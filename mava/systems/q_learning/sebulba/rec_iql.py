# Copyright 2022 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
import queue
import threading
import warnings
from collections import defaultdict
from queue import Queue
from typing import Any, Dict, List, Sequence, Tuple

import chex
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import optax
from colorama import Fore, Style
from flax.core.scope import FrozenVariableDict
from flax.linen import FrozenDict
from jax import Array, tree
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import get_sebulba_eval_fn as get_eval_fn
from mava.evaluator import make_rec_eval_act_fn
from mava.networks import RecQNetwork, ScannedRNN
from mava.systems.q_learning.types import Metrics, QNetParams, Transition
from mava.systems.q_learning.types import SebulbaLearnerState as LearnerState
from mava.types import Observation, SebulbaLearnerFn
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import base_sebulba_checks as check_sebulba_config
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import switch_leading_axes
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.sebulba.pipelines import OffPolicyPipeline as Pipeline
from mava.utils.sebulba.rate_limiters import BlockingRatioLimiter, RateLimiter, SampleToInsertRatio
from mava.utils.sebulba.utils import ParamsSource, RecordTimeTo, stop_sebulba
from mava.wrappers.episode_metrics import get_final_step_metrics
from mava.wrappers.gym import GymToJumanji


def rollout(
    key: chex.PRNGKey,
    env: GymToJumanji,
    config: DictConfig,
    rollout_pipeline: Pipeline,
    params_source: ParamsSource,
    q_net: RecQNetwork,
    actor_device: int,
    seeds: List[int],
    stop_event: threading.Event,
    actor_id: int,
) -> None:
    """Collects trajectories from the environment by running rollouts.

    Args:
        key: The PRNG key for stochasticity.
        env: The environment to interact with.
        config: Configuration settings for rollout and environment.
        rollout_pipeline: Pipeline for sending collected trajectories to the learner.
        params_source: Provides the latest network parameters from the learner.
        q_net: The Q-network.
        actor_device: Index of the actor device to use for rollout.
        seeds: Seeds for environment initialization.
        stop_event: used to inform this thread that it is time to stop.
        actor_id: Unique identifier for the actor.
    """
    name = threading.current_thread().name
    print(f"{Fore.BLUE}{Style.BRIGHT}Thread {name} started{Style.RESET_ALL}")
    num_agents = config.system.num_agents
    move_to_device = lambda x: jax.device_put(x, device=actor_device)

    @jax.jit
    def select_eps_greedy_action(
        params: FrozenDict,
        hidden_state: jax.Array,
        obs: Observation,
        term_or_trunc: Array,
        key: chex.PRNGKey,
        t: int,
    ) -> Tuple[Array, Array, int]:
        """Selects an action epsilon-greedily.

        Args:
            params: Network parameters.
            hidden_state: Current RNN hidden state.
            obs: Observation from the environment.
            term_or_trunc: Termination or truncation flag.
            key: PRNG key for sampling.
            t: Current timestep (used for epsilon decay).

        Returns:
            Tuple containing the chosen action, next hidden state, and updated timestep.
        """

        eps = jnp.maximum(
            config.system.eps_min, 1 - (t / config.system.eps_decay) * (1 - config.system.eps_min)
        )

        obs = tree.map(lambda x: x[jnp.newaxis, ...], obs)
        term_or_trunc = tree.map(lambda x: x[jnp.newaxis, ...], term_or_trunc)

        next_hidden_state, eps_greedy_dist = q_net.apply(
            params, hidden_state, (obs, term_or_trunc), eps
        )

        action = eps_greedy_dist.sample(seed=key).squeeze(0)  # (B, A)

        return action, next_hidden_state, t + config.arch.num_envs

    next_timestep = env.reset(seed=seeds)
    next_dones = next_timestep.last()[..., jnp.newaxis]

    # Initialise hidden states.
    hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )
    hstate_tpu = tree.map(move_to_device, hstate)
    step_count = 0

    # Loop till the desired num_updates is reached.
    while not stop_event.is_set():
        # Rollout
        episode_metrics: List[Dict] = []
        traj: List[Transition] = []
        actor_timings: Dict[str, List[float]] = defaultdict(list)
        with RecordTimeTo(actor_timings["rollout_time"]):
            for _ in range(config.system.rollout_length):
                with RecordTimeTo(actor_timings["get_params_time"]):
                    params = params_source.get()  # Get the latest parameters from the learner

                timestep = next_timestep
                obs_tpu = move_to_device(timestep.observation)

                dones = move_to_device(next_dones)

                # Get action and value
                with RecordTimeTo(actor_timings["compute_action_time"]):
                    key, act_key = jax.random.split(key)
                    action, hstate_tpu, step_count = select_eps_greedy_action(
                        params, hstate_tpu, obs_tpu, dones, act_key, step_count
                    )
                    cpu_action = jax.device_get(action)

                # Step environment
                with RecordTimeTo(actor_timings["env_step_time"]):
                    next_timestep = env.step(cpu_action)

                # Prepare the transation
                terminal = (1 - timestep.discount[..., 0, jnp.newaxis]).astype(bool)
                next_dones = next_timestep.last()[..., jnp.newaxis]

                # Append data to storage
                traj.append(
                    Transition(
                        timestep.observation,
                        action,
                        next_timestep.reward,
                        terminal,
                        next_dones,
                        next_timestep.extras["real_next_obs"],
                    )
                )

                episode_metrics.append(timestep.extras["episode_metrics"])

        # send trajectories to learner
        with RecordTimeTo(actor_timings["rollout_put_time"]):
            try:
                rollout_pipeline.put(traj, (actor_timings, episode_metrics), actor_id)
            except queue.Full:
                err = "Waited too long to add to the rollout queue, killing the actor thread"
                warnings.warn(err, stacklevel=2)
                break

    env.close()


def get_learner_step_fn(
    q_net: RecQNetwork,
    update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> SebulbaLearnerFn[LearnerState, Transition]:
    """Get the learner function."""

    def _update_step(
        learner_state: LearnerState,
        traj_batch: Transition,
    ) -> Tuple[LearnerState, Metrics]:
        """Performs a single network update.

        Calculates targets based on the input trajectories and updates the Q-network
        parameters accordingly.

        Args:
            learner_state: Current learner state.
            traj_batch: Batch of transitions for training.
        """

        def prep_inputs_to_scannedrnn(obs: Observation, term_or_trunc: chex.Array) -> chex.Array:
            """Prepares inputs for the ScannedRNN network.

            Switches leading axes of observations and termination/truncation flags to match the
            (T, B, ...) format expected by the RNN. The replay buffer outputs data in (B, T, ...)
            format.

            Args:
                obs: Observation data.
                term_or_trunc: Termination/truncation flags.

            Returns:
                Tuple containing the initial hidden state and the formatted input data.
            """
            hidden_state = ScannedRNN.initialize_carry(
                (obs.agents_view.shape[0], obs.agents_view.shape[2]),
                config.network.hidden_state_dim,
            )
            # the rb outputs (B, T, ... ) the RNN takes in (T, B, ...)
            obs = switch_leading_axes(obs)  # (B, T) -> (T, B)
            term_or_trunc = switch_leading_axes(term_or_trunc)  # (B, T) -> (T, B)
            obs_term_or_trunc = (obs, term_or_trunc)

            return hidden_state, obs_term_or_trunc

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def q_loss_fn(
                q_online_params: FrozenVariableDict,
                obs: Array,
                term_or_trunc: Array,
                action: Array,
                target: Array,
            ) -> Tuple[Array, Metrics]:
                # axes switched here to scan over time
                hidden_state, obs_term_or_trunc = prep_inputs_to_scannedrnn(obs, term_or_trunc)

                # get online q values of all actions
                _, q_online = q_net.apply(
                    q_online_params, hidden_state, obs_term_or_trunc, method="get_q_values"
                )
                q_online = switch_leading_axes(q_online)  # (T, B, ...) -> (B, T, ...)
                # get the q values of the taken actions and remove extra dim
                q_online = jnp.squeeze(
                    jnp.take_along_axis(q_online, action[..., jnp.newaxis], axis=-1), axis=-1
                )
                q_error = jnp.square(q_online - target)
                q_loss = jnp.mean(q_error)  # mse

                # pack metrics for logging
                loss_info = {
                    "q_loss": q_loss,
                    "mean_q": jnp.mean(q_online),
                    "mean_target": jnp.mean(target),
                }

                return q_loss, loss_info

            params, opt_states, t_train, traj_batch = update_state

            # Get data aligned with current/next timestep
            data_first = tree.map(lambda x: x[:, :-1, ...], traj_batch)
            data_next = tree.map(lambda x: x[:, 1:, ...], traj_batch)

            obs = data_first.obs
            term_or_trunc = data_first.term_or_trunc
            reward = data_first.reward
            action = data_first.action

            # The three following variables all come from the same time step.
            # They are stored and accessed in this way because of the `AutoResetWrapper`.
            # At the end of an episode `data_first.next_obs` and `data_next.obs` will be
            # different, which is why we need to store both. Thus `data_first.next_obs`
            # aligns with the `terminal` from `data_next`.
            next_obs = data_first.next_obs
            next_term_or_trunc = data_next.term_or_trunc
            next_terminal = data_next.terminal

            # Scan over each sample
            hidden_state, next_obs_term_or_trunc = prep_inputs_to_scannedrnn(
                next_obs, next_term_or_trunc
            )

            # eps defaults to 0
            _, next_online_greedy_dist = q_net.apply(
                params.online, hidden_state, next_obs_term_or_trunc
            )

            _, next_q_vals_target = q_net.apply(
                params.target, hidden_state, next_obs_term_or_trunc, method="get_q_values"
            )

            # Get the greedy action
            next_action = next_online_greedy_dist.mode()  # (T, B, ...)

            # Double q-value selection
            next_q_val = jnp.squeeze(
                jnp.take_along_axis(next_q_vals_target, next_action[..., jnp.newaxis], axis=-1),
                axis=-1,
            )

            next_q_val = switch_leading_axes(next_q_val)  # (T, B, ...) -> (B, T, ...)

            # TD Target
            target_q_val = reward + (1.0 - next_terminal) * config.system.gamma * next_q_val

            # Update Q function.
            q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
            q_grads, q_loss_info = q_grad_fn(
                params.online, obs, term_or_trunc, action, target_q_val
            )

            # Mean over the device and batch dimension.
            q_grads, q_loss_info = lax.pmean((q_grads, q_loss_info), axis_name="learner_devices")
            q_updates, next_opt_state = update_fn(q_grads, opt_states)
            next_online_params = optax.apply_updates(params.online, q_updates)

            if config.system.hard_update:
                next_target_params = optax.periodic_update(
                    next_online_params, params.target, t_train, config.system.update_period
                )
            else:
                next_target_params = optax.incremental_update(
                    next_online_params, params.target, config.system.tau
                )

            # Repack params and opt_states.
            next_params = QNetParams(next_online_params, next_target_params)

            # Repack.
            next_state = (next_params, next_opt_state, t_train + 1, traj_batch)

            return next_state, q_loss_info

        update_state = (*learner_state, traj_batch)
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, train_step, _ = update_state
        learner_state = LearnerState(params, opt_states, train_step)
        return learner_state, loss_info

    def learner_fn(
        learner_state: LearnerState, traj_batch: Transition
    ) -> Tuple[LearnerState, Metrics]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized across learner devices.

        Args:
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - step_counter int): Number of learning steps.
            traj_batch (Transition): The collected trainig data.
        """
        learner_state, loss_info = _update_step(learner_state, traj_batch)

        return learner_state, loss_info

    return learner_fn


def learner_thread(
    learn_fn: SebulbaLearnerFn[LearnerState, Transition],
    learner_state: LearnerState,
    config: DictConfig,
    eval_queue: Queue,
    pipeline: Pipeline,
    params_sources: Sequence[ParamsSource],
) -> None:
    for _ in range(config.arch.num_evaluation):
        # Create the lists to store metrics and timings for this learning iteration.
        ep_metrics_list: List[Dict] = []
        train_metrics: List[Dict] = []
        rollout_times_list: List[Dict] = []
        learn_times: Dict[str, List[float]] = defaultdict(list)

        with RecordTimeTo(learn_times["learner_time_per_eval"]):
            for _ in range(config.system.num_updates_per_eval):
                # Get the trajectory batch from the pipeline
                # This is blocking so it will wait until the pipeline has data.
                with RecordTimeTo(learn_times["rollout_get_time"]):
                    traj_batch, (rollout_time, ep_metric) = pipeline.get()  # type: ignore
                # Update the networks
                with RecordTimeTo(learn_times["learning_time"]):
                    learner_state, train_metric = learn_fn(learner_state, traj_batch)

                train_metrics.append(train_metric)
                if ep_metric is not None:
                    ep_metrics_list.append(ep_metric)
                    rollout_times_list.append(rollout_time)

                # Update all the params sources so all actors can get the latest params
                params = jax.block_until_ready(learner_state.params)
                for source in params_sources:
                    source.update(params.online)

        # Pass all the metrics and  params to the main thread (evaluator) for logging and evaluation

        if ep_metrics_list:
            # [{metric1 : (num_envs, ...), ...}] * n_rollouts -->
            # {metric1 : (n_rollouts, num_envs, ...), ...
            ep_metrics = tree.map(lambda *x: np.asarray(x), *ep_metrics_list)

            # [{metric1: value1, ...}] * n_rollouts -->
            # {metric1: mean(value1_rollout1, value1_rollout2, ...), ...}
            rollout_times = tree.map(lambda *x: np.mean(x), *rollout_times_list)
        else:
            rollout_times = {}
            ep_metrics = {}

        train_metrics = tree.map(lambda *x: np.asarray(x), *train_metrics)

        # learn_times : {metric1: (1,) or (num_updates_per_eval,), ...}
        time_metrics = rollout_times | learn_times
        # time_metrics  : {metric1: Array, ...} - > {metric1: mean(Array), ...}
        time_metrics = tree.map(np.mean, time_metrics, is_leaf=lambda x: isinstance(x, list))

        eval_queue.put((ep_metrics, train_metrics, learner_state, time_metrics))


def learner_setup(
    key: chex.PRNGKey, config: DictConfig, learner_devices: List
) -> Tuple[
    SebulbaLearnerFn[LearnerState, Transition],
    RecQNetwork,
    LearnerState,
    Sharding,
    Transition,
]:
    """Initialise learner_fn, network and learner state."""

    # create temporary environments.
    env = environments.make_gym_env(config, 1)
    # Get number of agents and actions.
    action_space = env.single_action_space
    config.system.num_agents = len(action_space)
    config.system.num_actions = int(action_space[0].n)

    devices = mesh_utils.create_device_mesh((len(learner_devices),), devices=learner_devices)
    mesh = Mesh(devices, axis_names=("learner_devices"))
    model_spec = PartitionSpec()
    data_spec = PartitionSpec("learner_devices")
    learner_sharding = NamedSharding(mesh, model_spec)

    key, q_key = jax.random.split(key, 2)
    # Shape legend:
    # T: Time (dummy dimension size = 1)
    # B: Batch (dummy dimension size = 1)
    # A: Agent
    # Make dummy inputs to init recurrent Q network -> need shape (T, B, A, ...)
    init_agents_view = jnp.array(env.single_observation_space.sample())
    init_action_mask = jnp.ones((config.system.num_agents, config.system.num_actions))
    init_obs = Observation(init_agents_view, init_action_mask)  # (A, ...)
    # (B, T, A, ...)
    init_obs_batched = tree.map(lambda x: x[jnp.newaxis, jnp.newaxis, ...], init_obs)
    dones = jnp.zeros((1, 1, 1), dtype=bool)  # (T, B, 1)
    init_x = (init_obs_batched, dones)  # pack the RNN dummy inputs
    # (B, A, ...)
    init_hidden_state = ScannedRNN.initialize_carry(
        (config.system.sample_batch_size, config.system.num_agents), config.network.hidden_state_dim
    )

    # Making recurrent Q network.
    pre_torso = hydra.utils.instantiate(config.network.q_network.pre_torso)
    post_torso = hydra.utils.instantiate(config.network.q_network.post_torso)
    q_net = RecQNetwork(
        pre_torso,
        post_torso,
        config.system.num_actions,
        config.network.hidden_state_dim,
    )
    q_params = q_net.init(q_key, init_hidden_state, init_x)  # epsilon defaults to 0
    q_target_params = q_net.init(q_key, init_hidden_state, init_x)  # ensure parameters are separate

    # Pack Q network params
    params = QNetParams(q_params, q_target_params)

    # Making optimiser and state
    opt = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(learning_rate=config.system.q_lr, eps=1e-5),
    )
    opt_state = opt.init(params.online)

    # Create dummy transition Used to initialiwe the pipeline's Buffer
    init_acts = env.single_action_space.sample()  # (A,)
    init_transition = Transition(
        obs=init_obs,  # (A, ...)
        action=init_acts,
        reward=jnp.zeros((config.system.num_agents,), dtype=float),
        terminal=jnp.zeros((1,), dtype=bool),  # one flag for all agents
        term_or_trunc=jnp.zeros((1,), dtype=bool),
        next_obs=init_obs,
    )

    learn_state_spec = LearnerState(model_spec, model_spec, model_spec)
    learn = get_learner_step_fn(q_net, opt.update, config)
    learn = jax.jit(
        shard_map(
            learn,
            mesh=mesh,
            in_specs=(learn_state_spec, data_spec),
            out_specs=(learn_state_spec, data_spec),
        )
    )

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        # Update the params
        params = restored_params

    # Duplicate learner across Learner devices.
    params, opt_state = jax.device_put((params, opt_state), learner_sharding)

    # Initial learner state.
    init_learner_state = LearnerState(params, opt_state, 0)

    env.close()
    return learn, q_net, init_learner_state, learner_sharding, init_transition


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    config = copy.deepcopy(_config)

    local_devices = jax.local_devices()
    devices = jax.devices()
    err = "Local and global devices must be the same, we dont support multihost yet"
    assert len(local_devices) == len(devices), err
    learner_devices = [devices[d_id] for d_id in config.arch.learner_device_ids]
    actor_devices = [local_devices[device_id] for device_id in config.arch.actor_device_ids]

    # JAX and numpy RNGs
    key = jax.random.PRNGKey(config.system.seed)
    np_rng = np.random.default_rng(config.system.seed)

    # Setup learner.
    learn, q_net, learner_state, learner_sharding, init_transition = learner_setup(
        key, config, learner_devices
    )

    # Setup evaluator.
    # One key per device for evaluation.
    eval_act_fn = make_rec_eval_act_fn(q_net.apply, config)

    evaluator, evaluator_envs = get_eval_fn(
        environments.make_gym_env, eval_act_fn, config, np_rng, absolute_metric=False
    )

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    check_sebulba_config(config)

    steps_per_rollout = (
        config.system.rollout_length
        * config.arch.num_envs
        * config.system.num_updates_per_eval
        * len(config.arch.actor_device_ids)
        * config.arch.n_threads_per_executor
    )

    # Setup RateLimiter
    # Replay_ratio = num_gradient_updates / num_env_steps
    # num_gradient_updates = sample_batch_size * epochs * rollout_length * samples_per_insert
    num_updates_per_insert = (
        config.system.epochs * config.system.sample_batch_size * config.system.rollout_length
    )
    num_setps_per_insert = (
        config.system.sample_sequence_length
        * config.arch.num_envs
        * len(config.arch.actor_device_ids)
        * config.arch.n_threads_per_executor
    )
    config.system.sample_per_insert = (
        num_setps_per_insert * config.system.replay_ratio
    ) / num_updates_per_insert

    min_num_inserts = max(
        config.system.sample_sequence_length // config.system.rollout_length,
        config.system.min_buffer_size // config.system.rollout_length,
        1,
    )

    rate_limiter: RateLimiter
    if config.system.error_tolerance:
        rate_limiter = SampleToInsertRatio(
            config.system.sample_per_insert, min_num_inserts, config.system.error_tolerance
        )
    else:
        rate_limiter = BlockingRatioLimiter(config.system.sample_per_insert, min_num_inserts)

    # Setup logger
    logger = MavaLogger(config)
    print_cfg: Dict = OmegaConf.to_container(config, resolve=True)
    print_cfg["arch"]["devices"] = jax.devices()
    pprint(print_cfg)

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Executor setup and launch.
    inital_params = jax.device_put(learner_state.params, actor_devices[0])  # unreplicate

    # Setup Pipeline
    pipe = Pipeline(config, learner_sharding, key, rate_limiter, init_transition)
    pipe.start()

    params_sources: List[ParamsSource] = []
    actor_threads: List[threading.Thread] = []
    actors_stop_event = threading.Event()
    # Create the actor threads
    print(f"{Fore.BLUE}{Style.BRIGHT}Starting up actor threads...{Style.RESET_ALL}")
    for device_idx, actor_device in enumerate(actor_devices):
        # Create 1 params source per device
        params_source = ParamsSource(inital_params.online, actor_device)
        params_source.start()
        params_sources.append(params_source)
        # Create multiple rollout threads per actor device
        for thread_id in range(config.arch.n_threads_per_executor):
            key, act_key = jax.random.split(key)
            seeds = np_rng.integers(np.iinfo(np.int32).max, size=config.arch.num_envs).tolist()
            act_key = jax.device_put(key, actor_device)
            actor_id = device_idx * config.arch.n_threads_per_executor + thread_id
            actor = threading.Thread(
                target=rollout,
                args=(
                    act_key,
                    # We have to do this here, creating envs inside actor threads causes deadlocks
                    environments.make_gym_env(config, config.arch.num_envs),
                    config,
                    pipe,
                    params_source,
                    q_net,
                    actor_device,
                    seeds,
                    actors_stop_event,
                    actor_id,
                ),
                name=f"Actor-{actor_device}-{thread_id}",
            )
            actor_threads.append(actor)

    # Start the actors simultaneously
    for actor in actor_threads:
        actor.start()

    eval_queue: Queue = Queue()
    threading.Thread(
        target=learner_thread,
        name="Learner",
        args=(learn, learner_state, config, eval_queue, pipe, params_sources),
    ).start()

    max_episode_return = -np.inf
    best_params_cpu = jax.device_get(inital_params.online)

    eval_hs = ScannedRNN.initialize_carry(
        (min(config.arch.num_eval_episodes, config.arch.num_envs), config.system.num_agents),
        config.network.hidden_state_dim,
    )

    # This is the main loop, all it does is evaluation and logging.
    # Acting and learning is happening in their own threads.
    # This loop waits for the learner to finish an update before evaluation and logging.
    for eval_step in range(config.arch.num_evaluation):
        # Sync with the learner - the get() is blocking so it keeps eval and learning in step.
        episode_metrics, train_metrics, learner_state, time_metrics = eval_queue.get()

        t = int(steps_per_rollout * (eval_step + 1))
        time_metrics |= {"timestep": t, "pipline_size": pipe.qsize()}
        logger.log(time_metrics, t, eval_step, LogEvent.MISC)

        if episode_metrics:
            episode_metrics, ep_completed = get_final_step_metrics(episode_metrics)
            episode_metrics["steps_per_second"] = steps_per_rollout / time_metrics["rollout_time"]
            if ep_completed:
                logger.log(episode_metrics, t, eval_step, LogEvent.ACT)

        train_metrics["learner_step"] = (eval_step + 1) * config.system.num_updates_per_eval
        train_metrics["learner_steps_per_second"] = (
            config.system.num_updates_per_eval
        ) / time_metrics["learner_time_per_eval"]
        logger.log(train_metrics, t, eval_step, LogEvent.TRAIN)

        learner_state_cpu = jax.device_get(learner_state)
        key, eval_key = jax.random.split(key, 2)
        eval_metrics = evaluator(
            learner_state_cpu.params.online, eval_key, {"hidden_state": eval_hs}
        )
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        episode_return = np.mean(eval_metrics["episode_return"])

        if save_checkpoint:  # Save a checkpoint of the learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=learner_state_cpu,
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and (max_episode_return <= episode_return):
            best_params_cpu = copy.deepcopy(learner_state_cpu.params.online)
            max_episode_return = float(episode_return)

    evaluator_envs.close()
    eval_performance = float(np.mean(eval_metrics[config.env.eval_metric]))

    # Gracefully shutting down all actors and resources.
    stop_sebulba(actors_stop_event, pipe, params_sources, actor_threads)

    # Measure absolute metric.
    if config.arch.absolute_metric:
        print(f"{Fore.BLUE}{Style.BRIGHT}Measuring absolute metric...{Style.RESET_ALL}")
        abs_metric_evaluator, abs_metric_evaluator_envs = get_eval_fn(
            environments.make_gym_env, eval_act_fn, config, np_rng, absolute_metric=True
        )
        key, eval_key = jax.random.split(key, 2)
        eval_hs = ScannedRNN.initialize_carry(
            (
                min(config.arch.num_absolute_metric_eval_episodes, config.arch.num_envs),
                config.system.num_agents,
            ),
            config.network.hidden_state_dim,
        )
        eval_metrics = abs_metric_evaluator(best_params_cpu, eval_key, {"hidden_state": eval_hs})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)
        abs_metric_evaluator_envs.close()

    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_iql_sebulba.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    cfg.logger.system_name = "rec_iql"

    # Run experiment.
    final_return = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}IDQN experiment completed{Style.RESET_ALL}")

    return float(final_return)


if __name__ == "__main__":
    hydra_entry_point()
