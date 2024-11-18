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
import time
import threading
import warnings
from collections import defaultdict
from typing import Any, Callable, Dict, Tuple, List, Sequence
from numpy.typing import NDArray
import queue
from queue import Queue

import chex
import flashbax as fbx
import hydra
import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import optax
from chex import PRNGKey
from colorama import Fore, Style
from flashbax.buffers.flat_buffer import TrajectoryBuffer
from flax.core.scope import FrozenVariableDict
from flax.linen import FrozenDict
from jax import Array, tree
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding
from jumanji.env import Environment
from jumanji.types import TimeStep
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import get_sebulba_eval_fn as get_eval_fn
from mava.evaluator import make_rec_eval_act_fn

from mava.networks import RecQNetwork, ScannedRNN
from mava.utils.sebulba import ParamsSource, OfflinePipeline as Pipeline, RecordTimeTo, ThreadLifetime, SampleToInsertRatio
from mava.systems.q_learning.types import (
    ActionSelectionState,
    ActionState,
    Metrics,
    QNetParams,
    TrainState,
    Transition,
)
from mava.systems.ppo.types import LearnerState #todo n0
from mava.types import Observation, SebulbaLearnerFn, ExperimentOutput 
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps, check_sebulba_config
from mava.utils.jax_utils import (
    switch_leading_axes,
    unreplicate_batch_dim,
    unreplicate_n_dims,
)
from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers.episode_metrics import get_final_step_metrics
from mava.wrappers.gym import GymToJumanji


def rollout(
    key: chex.PRNGKey,
    env: GymToJumanji,
    config: DictConfig,
    rollout_queue: Pipeline,
    params_source: ParamsSource,
    apply_fn ,
    actor_device: int,
    seeds: List[int],
    thread_lifetime: ThreadLifetime,
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
        thread_lifetime (ThreadLifetime): Manages the thread's lifecycle.
    """
    name = threading.current_thread().name
    print(f"{Fore.BLUE}{Style.BRIGHT}Thread {name} started{Style.RESET_ALL}")
    num_agents, num_envs = config.system.num_agents, config.arch.num_envs
    move_to_device = lambda x: jax.device_put(x, device=actor_device)

    @jax.jit
    def select_eps_greedy_action(
        params , hidden_state, obs: Observation, term_or_trunc: Array, key, t: int
    ) -> Tuple[ActionSelectionState, Array]: #todo
        """Select action to take in epsilon-greedy way. Batch and agent dims are included.

        Args:
        ----
            action_selection_state: Tuple of online parameters, previous hidden state,
                environment timestep (used to calculate epsilon) and a random key.
            obs: The observation from the previous timestep.
            term_or_trunc: The flag timestep.last() from the previous timestep.

        Returns:
        -------
            A tuple of the updated action selection state and the chosen action.

        """

        eps = jnp.maximum(
            config.system.eps_min, 1 - (t / config.system.eps_decay) * (1 - config.system.eps_min)
        )

        obs = tree.map(lambda x: x[jnp.newaxis, ...], obs)
        term_or_trunc = tree.map(lambda x: x[jnp.newaxis, ...], term_or_trunc)

        next_hidden_state, eps_greedy_dist = apply_fn(
            params.online, hidden_state, (obs, term_or_trunc), eps #todo only pass online params
        )

        action = eps_greedy_dist.sample(seed=key)
        action = action[0, ...]  # (1, B, A) -> (B, A)
        
        return action, next_hidden_state, t + config.arch.num_envs 

    next_timestep = env.reset(seed=seeds)
    dones = next_timestep.last()[..., jnp.newaxis]
    
    # Initialise hidden states.
    hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )
    hstate_tpu = tree.map(move_to_device, hstate)
    step_count = 0
    
    # Loop till the desired num_updates is reached.
    while not thread_lifetime.should_stop():
        # Rollout
        episode_metrics: List[Dict] = []
        traj: List[Transition] = []
        actor_timings: Dict[str, List[float]] = defaultdict(list)
        with RecordTimeTo(actor_timings["rollout_time"]):
            for _ in range(config.system.rollout_length):
                with RecordTimeTo(actor_timings["get_params_time"]):
                    params = params_source.get()  # Get the latest parameters from the learner
                
                timestep = next_timestep
                obs_tpu = tree.map(move_to_device, timestep.observation)
                
                last_dones = tree.map(move_to_device, dones)
                
                # Get action and value
                with RecordTimeTo(actor_timings["compute_action_time"]):
                    key, act_key = jax.random.split(key)
                    action, hstate_tpu, step_count = select_eps_greedy_action(params, hstate_tpu, obs_tpu, last_dones, act_key, step_count)
                    cpu_action = jax.device_get(action)

                # Step environment
                with RecordTimeTo(actor_timings["env_step_time"]):
                    next_timestep = env.step(cpu_action)

                #Prepare the transation
                terminal = (1 - timestep.discount[..., 0, jnp.newaxis]).astype(bool)
                dones = next_timestep.last()[..., jnp.newaxis] 

                # Append data to storage
                traj.append(
                    Transition(
                    timestep.observation, 
                    action, 
                    next_timestep.reward, 
                    terminal, 
                    dones, 
                    next_timestep.extras["real_next_obs"]
                    )
                )

                episode_metrics.append(timestep.extras["episode_metrics"])

        # send trajectories to learner
        with RecordTimeTo(actor_timings["rollout_put_time"]):
            try:
                rollout_queue.put(traj, (actor_timings, episode_metrics))
            except queue.Full:
                err = "Waited too long to add to the rollout queue, killing the actor thread"
                warnings.warn(err, stacklevel=2)
                break

    env.close()

#todo trainerState is elearner state in ppo, why did we switch up?

def get_learner_step_fn(
    apply_fn ,
    update_fn: optax.TransformUpdateFn,
    config: DictConfig,
) -> SebulbaLearnerFn[LearnerState, Transition]:
    """Get the learner function."""

    def _update_step(
        learner_state: TrainState,
        traj_batch: Transition,
    ) -> Tuple[LearnerState, Metrics]:
        """A single update of the network.

        This function calculates advantages and targets based on the trajectories
        from the actor and updates the actor and critic networks based on the losses.

        Args:
            learner_state (LearnerState): contains all the items needed for learning.
            traj_batch (PPOTransition): the batch of data to learn with.
        """
        
        
        def prep_inputs_to_scannedrnn(obs: Observation, term_or_trunc: chex.Array) -> chex.Array:
            """Prepares the inputs to the RNN network for either getting q values or the
            eps-greedy distribution.

            Mostly swaps leading axes because the replay buffer outputs (B, T, ... )
            and the RNN takes in (T, B, ...).
            """
            hidden_state = ScannedRNN.initialize_carry(
                (config.system.sample_batch_size, obs.agents_view.shape[2]), config.network.hidden_state_dim
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
                _, q_online = apply_fn(
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
            
            params, opt_states, traj_batch, t_train = update_state

 
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
            hidden_state, next_obs_term_or_trunc = prep_inputs_to_scannedrnn( #todo how/why are we re_init the hidden state each step? shoudn't be stored?
                next_obs, next_term_or_trunc
            )

            # eps defaults to 0
            _, next_online_greedy_dist = apply_fn(
                params.online, hidden_state, next_obs_term_or_trunc
            )

            _, next_q_vals_target = apply_fn(
                params.target, hidden_state, next_obs_term_or_trunc, method="get_q_values"
            )

            # Get the greedy action
            next_action = next_online_greedy_dist.mode()  # (T, B, ...)

            # Double q-value selection
            next_q_val = jnp.squeeze(
                jnp.take_along_axis(next_q_vals_target, next_action[..., jnp.newaxis], axis=-1), axis=-1
            )

            next_q_val = switch_leading_axes(next_q_val)  # (T, B, ...) -> (B, T, ...)

            # TD Target
            target_q_val = reward + (1.0 - next_terminal) * config.system.gamma * next_q_val

            # Update Q function.
            q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
            q_grads, q_loss_info = q_grad_fn(params.online, obs, term_or_trunc, action, target_q_val)

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
            next_state = (next_params, next_opt_state, traj_batch, t_train + 1)
  
            return next_state, q_loss_info

        
        #TODO this should be included in the learner state and inc by 1 each time we update )
        update_state = (learner_state.params, learner_state.opt_states, traj_batch, 0)
        # Update epochs #TODO BRODCsAT THE TRAJ_BATCH 
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.epochs
        )

        params, opt_states, traj_batch, train_step = update_state
        learner_state = LearnerState(params, opt_states, None, None, learner_state.timestep)
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
        # the first axis to be time #todo is this comment still relevent ? 
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
        rollout_times: List[Dict] = []
        learn_times: Dict[str, List[float]] = defaultdict(list)

        with RecordTimeTo(learn_times["learner_time_per_eval"]):
            for _ in range(config.system.num_updates_per_eval):
                # Get the trajectory batch from the pipeline
                # This is blocking so it will wait until the pipeline has data.
                with RecordTimeTo(learn_times["rollout_get_time"]):
                    traj_batch, (rollout_time, ep_metrics) = pipeline.get(block=True)

                # Update the networks
                with RecordTimeTo(learn_times["learning_time"]):
                    learner_state, train_metrics = learn_fn(learner_state, traj_batch)

                metrics.append((ep_metrics, train_metrics))
                rollout_times.append(rollout_time)

                # Update all the params sources so all actors can get the latest params
                params = jax.block_until_ready(learner_state.params)
                for source in params_sources:
                    source.update(params)

        # Pass all the metrics and  params to the main thread (evaluator) for logging and evaluation
        ep_metrics, train_metrics = tree.map(lambda *x: np.asarray(x), *metrics)
        rollout_times: Dict[str, NDArray] = tree.map(lambda *x: np.mean(x), *rollout_times)
        timing_dict = rollout_times | learn_times
        timing_dict = tree.map(np.mean, timing_dict, is_leaf=lambda x: isinstance(x, list))

        eval_queue.put((ep_metrics, train_metrics, learner_state, timing_dict))

def learner_setup(
    key: chex.PRNGKey, config: DictConfig, learner_devices: List
) -> Tuple[
    SebulbaLearnerFn[LearnerState, Transition],
    RecQNetwork,
    LearnerState,
    Sharding,
]:
    """Initialise learner_fn, network and learner state."""

    # create temporory envoirnments.
    env = environments.make_gym_env(config, 1)
    # Get number of agents and actions.
    action_space = env.single_action_space
    config.system.num_agents = len(action_space)
    config.system.num_actions = int(action_space[0].n)
    
    devices = mesh_utils.create_device_mesh((len(learner_devices), ), devices=learner_devices)
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
    init_obs = Observation(init_agents_view, init_action_mask) # (A, ...)
    # (B, T, A, ...)
    init_obs_batched = tree.map(lambda x: x[jnp.newaxis, jnp.newaxis, ...], init_obs)
    dones = jnp.zeros((1, 1, 1), dtype=bool)  # (T, B, 1)
    init_x = (init_obs_batched, dones)  # pack the RNN dummy inputs
    # (B, A, ...)
    init_hidden_state = ScannedRNN.initialize_carry(
        (config.arch.num_envs, config.system.num_agents), config.network.hidden_state_dim
    )

    # Making recurrent Q network.shao
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

    # Create dummy transition
    init_acts = env.single_action_space.sample()  # (A,)
    init_transition = Transition(
        obs=init_obs,  # (A, ...)
        action=init_acts,
        reward=jnp.zeros((config.system.num_agents,), dtype=float),
        terminal=jnp.zeros((1,), dtype=bool),  # one flag for all agents
        term_or_trunc=jnp.zeros((1,), dtype=bool),
        next_obs=init_obs
    )

    # Initialise trajectory buffer
    rb = fbx.make_trajectory_buffer(
        # n transitions gives n-1 full data points
        sample_sequence_length=config.system.sample_sequence_length + 1,
        period=1,  # sample any unique trajectory
        add_batch_size=config.arch.num_envs,
        sample_batch_size=config.system.sample_batch_size,
        max_length_time_axis=config.system.buffer_size,
        min_length_time_axis=config.system.min_buffer_size,
    )
    buffer_state = rb.init(init_transition)
    
    learn_state_spec = LearnerState(model_spec, model_spec, data_spec, None, data_spec)
    learn = get_learner_step_fn(q_net.apply, opt.update, config)
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
    params, opt_state, step_keys = jax.device_put(
        (params, opt_state, step_keys), learner_sharding
    )
    
    # Initial learner state.
    init_learner_state = LearnerState(
        params, opt_state, step_keys, None, None
    )
    
    env.close()
    return learn, q_net.apply, init_learner_state, learner_sharding , rb.add, rb.sample, buffer_state


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
    learn, apply_fn, learner_state, learner_sharding, buffer_add, buffer_sample, buffer_state = learner_setup(key, config, learner_devices)

    # Setup evaluator.
    # One key per device for evaluation.
    eval_act_fn = make_rec_eval_act_fn(apply_fn, config)
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

    # the rollout queue/ the pipe between actor and learner
    pipe_lifetime = ThreadLifetime()
    
    # Calculate the replay ratio todo: refactor the configs
    steps_per_insert = config.arch.num_envs * config.system.rollout_length
    samples_per_inserted_batched_rollout = config.system.epochs
    replay_ratio = samples_per_inserted_batched_rollout / steps_per_insert
    config.system.replay_ratio = replay_ratio
    # Set up the rate limiter that controls how actors and learners interact with the pipeline
    samples_per_insert_tolerance_rate = 0.1  # This allows for 10% tolerance
    samples_per_insert_tolerance = samples_per_insert_tolerance_rate * config.system.epochs
    error_buffer = config.system.sample_batch_size * samples_per_insert_tolerance
    min_inserts = max(config.system.sample_batch_size // steps_per_insert, 1)
    rate_limiter = SampleToInsertRatio(config.system.epochs, min_inserts, error_buffer)
    
    pipe = Pipeline(config.arch.rollout_queue_size, learner_sharding, pipe_lifetime, buffer_add, buffer_sample, buffer_state , key, rate_limiter)#todo chek key
    pipe.start()

    params_sources: List[ParamsSource] = []
    actor_threads: List[threading.Thread] = []
    actor_lifetime = ThreadLifetime()
    params_sources_lifetime = ThreadLifetime()

    # Create the actor threads
    print(f"{Fore.BLUE}{Style.BRIGHT}Starting up actor threads...{Style.RESET_ALL}")
    for actor_device in actor_devices:
        # Create 1 params source per device
        params_source = ParamsSource(inital_params, actor_device, params_sources_lifetime)
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
                    apply_fn,
                    actor_device,
                    seeds,
                    actor_lifetime,
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
        eval_metrics = evaluator(learner_state_cpu.params.online, eval_key, {"hidden_state" : eval_hs})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        episode_return = np.mean(eval_metrics["episode_return"])

        if save_checkpoint:  # Save a checkpoint of the learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=learner_state_cpu,
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params_cpu = copy.deepcopy(learner_state_cpu.params.online)
            max_episode_return = float(episode_return)

    evaluator_envs.close()
    eval_performance = float(np.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        print(f"{Fore.BLUE}{Style.BRIGHT}Measuring absolute metric...{Style.RESET_ALL}")
        abs_metric_evaluator, abs_metric_evaluator_envs = get_eval_fn(
            environments.make_gym_env, eval_act_fn, config, np_rng, absolute_metric=True
        )
        key, eval_key = jax.random.split(key, 2)
        eval_hs = ScannedRNN.initialize_carry(
            (min(config.arch.num_absolute_metric_eval_episodes, config.arch.num_envs), config.system.num_agents),
            config.network.hidden_state_dim,
        )
        eval_metrics = abs_metric_evaluator(best_params_cpu, eval_key, {"hidden_state" : eval_hs})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)
        abs_metric_evaluator_envs.close()

    # Stop all the threads.
    logger.stop()
    actor_lifetime.stop()
    pipe.clear()  # We clear the pipeline before stopping the actor threads to avoid deadlock
    print(f"{Fore.RED}{Style.BRIGHT}Pipe cleared{Style.RESET_ALL}")
    print(f"{Fore.RED}{Style.BRIGHT}Stopping actor threads...{Style.RESET_ALL}")
    for actor in actor_threads:
        actor.join()
        print(f"{Fore.RED}{Style.BRIGHT}{actor.name} stopped{Style.RESET_ALL}")
    print(f"{Fore.RED}{Style.BRIGHT}Stopping pipeline...{Style.RESET_ALL}")
    pipe_lifetime.stop()
    pipe.join()
    print(f"{Fore.RED}{Style.BRIGHT}Stopping params sources...{Style.RESET_ALL}")
    params_sources_lifetime.stop()
    for params_source in params_sources:
        params_source.join()
    print(f"{Fore.RED}{Style.BRIGHT}All threads stopped...{Style.RESET_ALL}")

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
