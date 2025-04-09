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
from functools import partial
from queue import Queue
from typing import Any, Callable, Dict, List, Sequence, Tuple

import chex
import hydra
import jax
import jax.debug
import jax.numpy as jnp
import numpy as np
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict as Params
from jax import tree
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import ActorState, EvalActFn, get_num_eval_envs
from mava.evaluator import get_sebulba_eval_fn as get_eval_fn
from mava.networks import SableNetwork
from mava.networks.utils.sable import get_init_hidden_state
from mava.systems.ppo.types import PPOTransition as Transition
from mava.systems.ppo.types import SebulbaLearnerState as LearnerState
from mava.systems.sable.types import ActorApply, LearnerApply
from mava.types import (
    Action,
    Metrics,
    Observation,
    SebulbaLearnerFn,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.config import ppo_sebulba_checks as check_sebulba_config
from mava.utils.jax_utils import merge_leading_dims, switch_leading_axes
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.sebulba.pipelines import Pipeline
from mava.utils.sebulba.utils import ParamsSource, RecordTimeTo, stop_sebulba
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics
from mava.wrappers.gym import GymToJumanji, TimeStep


def rollout(
    key: chex.PRNGKey,
    env: GymToJumanji,
    config: DictConfig,
    rollout_queue: Pipeline,
    params_source: ParamsSource,
    apply_fns: ActorApply,
    actor_device: int,
    seeds: List[int],
    stop_event: threading.Event,
) -> None:
    """Runs rollouts to collect trajectories from the environment.

    Args:
        key (chex.PRNGKey): The PRNGkey.
        config (DictConfig): Configuration settings for the environment and rollout.
        rollout_queue (Pipeline): Queue for sending collected rollouts to the learner.
        params_source (ParamsSource): Source for fetching the latest network parameters
        from the learner.
        apply_fns (Tuple): Functions for running the actor and critic networks.
        actor_device (Device): Actor device to use for rollout.
        seeds (List[int]): Seeds for initializing the environment.
        stop_event (threading.Event): Manages the thread's lifecycle.
    """
    name = threading.current_thread().name
    print(f"{Fore.BLUE}{Style.BRIGHT}Thread {name} started{Style.RESET_ALL}")

    num_agents, num_envs = config.system.num_agents, config.arch.num_envs
    move_to_device = lambda x: jax.device_put(x, device=actor_device)

    @jax.jit
    def act_fn(
        params: Params,
        observation: Observation,
        key: chex.PRNGKey,
    ) -> Tuple:
        """Get action and value."""
        action, log_prob, value, _ = apply_fns(  # type: ignore
            params,
            observation=observation,
            key=key,
            hstates=get_init_hidden_state(config.network.net_config, num_envs),
        )
        return action, log_prob, value

    timestep = env.reset(seed=seeds)
    dones = np.repeat(timestep.last(), num_agents).reshape(num_envs, -1)

    # Loop till the desired num_updates is reached.
    while not stop_event.is_set():
        # Rollout
        traj: List[Transition] = []
        episode_metrics: List[Dict] = []
        actor_timings: Dict[str, List[float]] = defaultdict(list)
        with RecordTimeTo(actor_timings["rollout_time"]):
            for _ in range(config.system.rollout_length):
                with RecordTimeTo(actor_timings["get_params_time"]):
                    params = params_source.get()  # Get the latest parameters from the learner

                obs_tpu = tree.map(move_to_device, timestep.observation)

                # Get action and value
                with RecordTimeTo(actor_timings["compute_action_time"]):
                    key, act_key = jax.random.split(key)
                    action, log_prob, value = act_fn(params, obs_tpu, act_key)
                    cpu_action = jax.device_get(action)

                # Step environment
                with RecordTimeTo(actor_timings["env_step_time"]):
                    timestep = env.step(cpu_action)

                # Append data to storage
                traj.append(Transition(dones, action, value, timestep.reward, log_prob, obs_tpu))
                dones = np.repeat(timestep.last(), num_agents).reshape(num_envs, -1)

                episode_metrics.append(timestep.extras["episode_metrics"])

        # send trajectories to learner
        with RecordTimeTo(actor_timings["rollout_put_time"]):
            try:
                rollout_queue.put(traj, (actor_timings, episode_metrics), (timestep, None))
            except queue.Full:
                err = "Waited too long to add to the rollout queue, killing the actor thread"
                warnings.warn(err, stacklevel=2)
                break

    env.close()


def get_learner_step_fn(
    apply_fns: Tuple[ActorApply, LearnerApply],
    update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> SebulbaLearnerFn[LearnerState, Transition]:
    """Get the learner function."""

    num_envs = config.arch.num_envs
    num_learner_envs = int(num_envs // len(config.arch.learner_device_ids))

    # Get apply and update functions for actor and critic networks.
    sable_action_select_fn, sable_apply_fn = apply_fns

    def _update_step(
        learner_state: LearnerState,
        traj_batch: Transition,
    ) -> Tuple[LearnerState, Metrics]:
        """A single update of the network.

        This function calculates advantages and targets based on the trajectories
        from the actor and updates the actor and critic networks based on the losses.

        Args:
            learner_state (LearnerState): contains all the items needed for learning.
            traj_batch (PPOTransition): the batch of data to learn with.
        """

        # Calculate advantage
        params, opt_states, key, _, final_timestep = learner_state
        key = jnp.squeeze(key, axis=0)
        _, _, last_val, _ = sable_action_select_fn(  # type: ignore
            params,
            observation=final_timestep.observation,
            key=key,
            hstates=get_init_hidden_state(config.network.net_config, num_learner_envs),
        )
        last_done = jnp.repeat(final_timestep.last(), config.system.num_agents).reshape(
            num_learner_envs, -1
        )
        advantages, targets = calculate_gae(
            traj_batch, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                params, opt_state, key = train_state
                traj_batch, advantages, targets = batch_info

                def _loss_fn(
                    params: Params,
                    traj_batch: Transition,
                    gae: chex.Array,
                    value_targets: chex.Array,
                    rng_key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate Sable loss."""
                    # Rerun network
                    minibatch_size = traj_batch.action.shape[0]
                    value, log_prob, entropy = sable_apply_fn(  # type: ignore
                        params,
                        observation=traj_batch.obs,
                        action=traj_batch.action,
                        dones=traj_batch.done,
                        rng_key=rng_key,
                        hstates=get_init_hidden_state(config.network.net_config, minibatch_size),
                    )

                    # Calculate actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    # Nomalise advantage at minibatch level
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    actor_loss1 = ratio * gae
                    actor_loss2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        )
                        * gae
                    )
                    actor_loss = -jnp.minimum(actor_loss1, actor_loss2)
                    actor_loss = actor_loss.mean()
                    entropy = entropy.mean()

                    # Clipped MSE loss
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - value_targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - value_targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_loss = (
                        actor_loss
                        - config.system.ent_coef * entropy
                        + config.system.vf_coef * value_loss
                    )
                    return total_loss, (actor_loss, entropy, value_loss)

                # CALCULATE ACTOR LOSS
                key, entropy_key = jax.random.split(key)
                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                loss_info, grads = grad_fn(params, traj_batch, advantages, targets, entropy_key)

                # Compute the parallel mean (pmean) over the learner devices.
                grads, loss_info = jax.lax.pmean((grads, loss_info), axis_name="learner_devices")

                # Update params and optimiser state
                updates, new_opt_state = update_fn(grads, opt_state)
                new_params = optax.apply_updates(params, updates)

                total_loss, (actor_loss, entropy, value_loss) = loss_info
                loss_info = {
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }

                return (new_params, new_opt_state, key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, agent_shuffle_key, entropy_key = jax.random.split(key, 4)

            # Shuffle minibatches
            batch_size = config.system.rollout_length * num_learner_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = tree.map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)

            # Shuffle agents
            agent_perm = jax.random.permutation(agent_shuffle_key, config.system.num_agents)
            shuffled_batch = tree.map(lambda x: jnp.take(x, agent_perm, axis=1), shuffled_batch)

            minibatches = tree.map(
                lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                shuffled_batch,
            )
            # Update minibatches
            (params, opt_states, _), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (params, opt_states, traj_batch, advantages, targets, key)
            return update_state, loss_info

        update_state = (params, opt_states, traj_batch, advantages, targets, key)
        # Update epochs
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        key = jnp.expand_dims(key, axis=0)  # add the learner_devices axis for shape consitency
        learner_state = LearnerState(params, opt_states, key, None, learner_state.timestep)
        return learner_state, loss_info

    def learner_fn(
        learner_state: LearnerState, traj_batch: Transition
    ) -> Tuple[LearnerState, Metrics]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The last timestep of the rollout.
        """
        # This function is shard mapped on the batch axis, but `_update_step` needs
        # the first axis to be time
        traj_batch = tree.map(switch_leading_axes, traj_batch)
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
        metrics: List[Tuple[Dict, Dict]] = []
        rollout_times_array: List[Dict] = []
        learn_times: Dict[str, List[float]] = defaultdict(list)

        with RecordTimeTo(learn_times["learner_time_per_eval"]):
            for _ in range(config.system.num_updates_per_eval):
                # Get the trajectory batch from the pipeline
                # This is blocking so it will wait until the pipeline has data.
                with RecordTimeTo(learn_times["rollout_get_time"]):
                    traj_batch, rollout_time, ep_metrics, (timestep, _) = pipeline.get(block=True)  # type: ignore

                # Replace the timestep in the learner state with the latest timestep
                # This means the learner has access to the entire trajectory as well as
                # an additional timestep which it can use to bootstrap.
                learner_state = learner_state._replace(timestep=timestep)
                # Update the networks
                with RecordTimeTo(learn_times["learning_time"]):
                    learner_state, train_metrics = learn_fn(learner_state, traj_batch)

                metrics.append((ep_metrics, train_metrics))
                rollout_times_array.append(rollout_time)

                # Update all the params sources so all actors can get the latest params
                params = jax.block_until_ready(learner_state.params)
                for source in params_sources:
                    source.update(params)

        # Pass all the metrics and  params to the main thread (evaluator) for logging and evaluation
        ep_metrics, train_metrics = tree.map(lambda *x: np.asarray(x), *metrics)
        rollout_times: Dict[str, NDArray] = tree.map(lambda *x: np.mean(x), *rollout_times_array)
        timing_dict = rollout_times | learn_times
        timing_dict = tree.map(np.mean, timing_dict, is_leaf=lambda x: isinstance(x, list))

        eval_queue.put((ep_metrics, train_metrics, learner_state, timing_dict))


def learner_setup(
    key: chex.PRNGKey, config: DictConfig, learner_devices: List
) -> Tuple[
    SebulbaLearnerFn[LearnerState, Transition],
    Callable,
    LearnerState,
    Sharding,
]:
    """Initialise learner_fn, network and learner state."""

    # create temporory envoirnments.
    env = environments.make_gym_env(config, config.arch.num_envs)
    # Get number of agents and actions.
    action_space = env.single_action_space
    config.system.num_agents = len(action_space)
    config.system.num_actions = int(action_space[0].n)

    devices = mesh_utils.create_device_mesh((len(learner_devices),), devices=learner_devices)
    mesh = Mesh(devices, axis_names=("learner_devices",))
    model_spec = PartitionSpec()
    data_spec = PartitionSpec("learner_devices")
    learner_sharding = NamedSharding(mesh, model_spec)

    # PRNG keys.
    key, net_key = jax.random.split(key, 2)

    # Setting the chunksize - many agent problems require chunking agents
    # Create a dummy decay factor for FF Sable
    config.network.memory_config.decay_scaling_factor = 1.0
    if config.network.memory_config.agents_chunk_size:
        config.network.memory_config.chunk_size = config.network.memory_config.agents_chunk_size
        err = "Number of agents should be divisible by chunk size"
        assert config.system.num_agents % config.network.memory_config.chunk_size == 0, err
    else:
        config.network.memory_config.chunk_size = config.system.num_agents

    # Set positional encoding to False, since ff-sable does not use temporal dependencies.
    config.network.memory_config.timestep_positional_encoding = False

    _, action_space_type = get_action_head(env.single_action_space)

    # Define network.
    sable_network = SableNetwork(
        n_agents=config.system.num_agents,
        n_agents_per_chunk=config.network.memory_config.chunk_size,
        action_dim=config.system.num_actions,
        net_config=config.network.net_config,
        memory_config=config.network.memory_config,
        action_space_type=action_space_type,
    )

    # Define optimiser.
    lr = make_learning_rate(config.system.actor_lr, config)
    optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(lr, eps=1e-5),
    )

    # Initialise observation: Select only obs for a single agent.
    init_obs = jnp.array(env.single_observation_space.sample()["agents_view"])
    init_action_mask = jnp.ones((config.system.num_agents, config.system.num_actions))
    step_count = jnp.zeros((config.system.num_agents))
    init_x = Observation(init_obs, init_action_mask, step_count)
    init_x = tree.map(lambda x: x[jnp.newaxis, ...], init_x)  # Add batch dim

    init_hs = get_init_hidden_state(config.network.net_config, config.arch.num_envs)
    init_hs = tree.map(lambda x: x[0, jnp.newaxis], init_hs)

    # Initialise params and optimiser state.
    params = sable_network.init(
        net_key,
        init_x,
        init_hs,
        net_key,
        method="get_actions",
    )
    opt_state = optim.init(params)

    # Pack apply and update functions.
    net_act_fn = partial(sable_network.apply, method="get_actions")
    net_learn_fn = partial(sable_network.apply)  # Training function
    apply_fns = (net_act_fn, net_learn_fn)

    # defines how the learner state is sharded: params, opt and key = sharded, timestep = sharded
    learn_state_spec = LearnerState(model_spec, model_spec, data_spec, None, data_spec)
    learn = get_learner_step_fn(apply_fns, optim.update, config)
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

    # Define params to be replicated across devices and batches.
    key, *step_keys = jax.random.split(key, len(learner_devices) + 1)
    step_keys = jnp.stack(step_keys, 0)

    # Duplicate learner across Learner devices.
    params, opt_state, step_keys = jax.device_put((params, opt_state, step_keys), learner_sharding)

    # Initialise learner state.
    init_learner_state = LearnerState(params, opt_state, step_keys, None, None)  # type: ignore
    env.close()

    return learn, net_act_fn, init_learner_state, learner_sharding  # type: ignore


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "ff_sable_sebulba"
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
    learn, select_action_fn, learner_state, learner_sharding = learner_setup(
        key, config, learner_devices
    )

    # Setup evaluator.
    def make_ff_sable_act_fn(actor_apply_fn: ActorApply) -> EvalActFn:
        def eval_act_fn(
            params: Params, timestep: TimeStep, key: chex.PRNGKey, actor_state: ActorState
        ) -> Tuple[Action, Dict]:
            output_action, _, _, _ = actor_apply_fn(  # type: ignore
                params,
                observation=timestep.observation,
                key=key,
            )
            return output_action, {}

        return eval_act_fn

    eval_batch_size = get_num_eval_envs(config, absolute_metric=False)
    eval_hs = get_init_hidden_state(config.network.net_config, eval_batch_size)
    sable_execution_fn = partial(select_action_fn, hstates=eval_hs)
    eval_act_fn = make_ff_sable_act_fn(sable_execution_fn)
    evaluator, evaluator_envs = get_eval_fn(
        environments.make_gym_env, eval_act_fn, config, np_rng, absolute_metric=False
    )

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    check_sebulba_config(config)

    steps_per_rollout = (
        config.system.rollout_length * config.arch.num_envs * config.system.num_updates_per_eval
    )

    # Logger setup
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

    # The rollout queue/ pipe between actor and learner
    pipe = Pipeline(config.arch.rollout_queue_size, learner_sharding)
    pipe.start()

    params_sources: List[ParamsSource] = []
    actor_threads: List[threading.Thread] = []
    actors_stop_event = threading.Event()

    # Create the actor threads
    print(f"{Fore.BLUE}{Style.BRIGHT}Starting up actor threads...{Style.RESET_ALL}")
    for actor_device in actor_devices:
        # Create 1 params source per device
        params_source = ParamsSource(inital_params, actor_device)
        params_source.start()
        params_sources.append(params_source)
        # Create multiple rollout threads per actor device
        for thread_id in range(config.arch.n_threads_per_executor):
            key, act_key = jax.random.split(key)
            seeds = np_rng.integers(np.iinfo(np.int32).max, size=config.arch.num_envs).tolist()
            act_key = jax.device_put(key, actor_device)

            actor = threading.Thread(
                target=rollout,
                args=(
                    act_key,
                    # We have to do this here, creating envs inside actor threads causes deadlocks
                    environments.make_gym_env(config, config.arch.num_envs),
                    config,
                    pipe,
                    params_source,
                    select_action_fn,
                    actor_device,
                    seeds,
                    actors_stop_event,
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
    best_params_cpu = jax.device_get(inital_params)

    # This is the main loop, all it does is evaluation and logging.
    # Acting and learning is happening in their own threads.
    # This loop waits for the learner to finish an update before evaluation and logging.
    for eval_step in range(config.arch.num_evaluation):
        # Sync with the learner - the get() is blocking so it keeps eval and learning in step.
        episode_metrics, train_metrics, learner_state, time_metrics = eval_queue.get()

        t = int(steps_per_rollout * (eval_step + 1))
        time_metrics |= {"timestep": t, "pipline_size": pipe.qsize()}
        logger.log(time_metrics, t, eval_step, LogEvent.MISC)

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
        eval_metrics = evaluator(learner_state_cpu.params, eval_key, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        episode_return = np.mean(eval_metrics["episode_return"])

        if save_checkpoint:  # Save a checkpoint of the learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=learner_state_cpu,
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params_cpu = copy.deepcopy(learner_state_cpu.params)
            max_episode_return = float(episode_return)

    evaluator_envs.close()
    eval_performance = float(np.mean(eval_metrics[config.env.eval_metric]))

    # Gracefully shutting down all actors and resources.
    stop_sebulba(actors_stop_event, pipe, params_sources, actor_threads)

    # Measure absolute metric.
    if config.arch.absolute_metric:
        print(f"{Fore.BLUE}{Style.BRIGHT}Measuring absolute metric...{Style.RESET_ALL}")
        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        abs_hs = get_init_hidden_state(config.network.net_config, eval_batch_size)
        sable_execution_fn = partial(select_action_fn, hstates=abs_hs)
        eval_act_fn = make_ff_sable_act_fn(sable_execution_fn)
        print(f"{Fore.BLUE}{Style.BRIGHT}Measuring absolute metric...{Style.RESET_ALL}")
        abs_metric_evaluator, abs_metric_evaluator_envs = get_eval_fn(
            environments.make_gym_env, eval_act_fn, config, np_rng, absolute_metric=True
        )
        key, eval_key = jax.random.split(key, 2)
        eval_metrics = abs_metric_evaluator(best_params_cpu, eval_key, {})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)
        abs_metric_evaluator_envs.close()

    # Stop all the threads.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default/",
    config_name="ff_sable_sebulba.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    cfg.logger.system_name = "ff_sable_sebulba"

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}FF Sable experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
