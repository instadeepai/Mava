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
import jax.debug
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jax import tree
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, Sharding
from jax.sharding import PartitionSpec as P
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import get_sebulba_eval_fn as get_eval_fn
from mava.evaluator import make_ff_eval_act_fn
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardValueNet as Critic
from mava.systems.ppo.types import LearnerState, OptStates, Params, PPOTransition
from mava.types import (
    ActorApply,
    CriticApply,
    ExperimentOutput,
    Observation,
    SebulbaLearnerFn,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_sebulba_config, check_total_timesteps
from mava.utils.jax_utils import merge_leading_dims, switch_leading_axes
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.sebulba import ParamsSource, Pipeline, RecordTimeTo, ThreadLifetime
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics
from mava.wrappers.gym import GymToJumanji


def rollout(
    key: chex.PRNGKey,
    env: GymToJumanji,
    config: DictConfig,
    rollout_queue: Pipeline,
    params_source: ParamsSource,
    apply_fns: Tuple[ActorApply, CriticApply],
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
    actor_apply_fn, critic_apply_fn = apply_fns
    num_agents, num_envs = config.system.num_agents, config.arch.num_envs
    move_to_device = lambda x: jax.device_put(x, device=actor_device)

    @jax.jit
    def act_fn(
        params: Params,
        observation: Observation,
        key: chex.PRNGKey,
    ) -> Tuple:
        """Get action and value."""
        actor_policy = actor_apply_fn(params.actor_params, observation)
        action = actor_policy.sample(seed=key)
        log_prob = actor_policy.log_prob(action)
        # It may be faster to calculate the values in the learner as
        # then we won't need to pass critic params to actors.
        value = critic_apply_fn(params.critic_params, observation).squeeze()
        return action, log_prob, value

    timestep = env.reset(seed=seeds)
    dones = np.repeat(timestep.last(), num_agents).reshape(num_envs, -1)

    # Loop till the desired num_updates is reached.
    while not thread_lifetime.should_stop():
        # Rollout
        traj: List[PPOTransition] = []
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
                    timestep = env.step(cpu_action.swapaxes(0, 1))

                dones = np.repeat(timestep.last(), num_agents).reshape(num_envs, -1)

                # Append data to storage
                traj.append(
                    PPOTransition(
                        dones,
                        action,
                        value,
                        timestep.reward,
                        log_prob,
                        obs_tpu,
                        timestep.extras,
                    )
                )

        # send trajectories to learner
        with RecordTimeTo(actor_timings["rollout_put_time"]):
            try:
                rollout_queue.put(traj, timestep, actor_timings)
            except queue.Full:
                err = "Waited too long to add to the rollout queue, killing the actor thread"
                warnings.warn(err, stacklevel=2)
                break

    env.close()


def get_learner_step_fn(
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> SebulbaLearnerFn[LearnerState, PPOTransition]:
    """Get the learner function."""

    num_agents, num_envs = config.system.num_agents, config.arch.num_envs
    num_learner_envs = int(num_envs // len(config.arch.learner_device_ids))

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state: LearnerState,
        traj_batch: PPOTransition,
    ) -> Tuple[LearnerState, Tuple]:
        """A single update of the network.

        This function calculates advantages and targets based on the trajectories
        from the actor and updates the actor and critic networks based on the losses.

        Args:
            learner_state (LearnerState): contains all the items needed for learning.
            traj_batch (PPOTransition): the batch of data to learn with.
        """

        def _calculate_gae(
            traj_batch: PPOTransition, last_val: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
            """Calculate the GAE."""

            gamma, gae_lambda = config.system.gamma, config.system.gae_lambda

            def _get_advantages(gae_and_next_value: Tuple, transition: PPOTransition) -> Tuple:
                """Calculate the GAE for a single transition."""
                gae, next_value = gae_and_next_value
                done, value, reward = transition.done, transition.value, transition.reward

                delta = reward + gamma * next_value * (1 - done) - value
                gae = delta + gamma * gae_lambda * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        # Calculate advantage
        params, opt_states, key, _, final_timestep = learner_state
        last_val = critic_apply_fn(params.critic_params, final_timestep.observation)
        advantages, targets = _calculate_gae(traj_batch, last_val)

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""

                # Unpack train state and batch info
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: PPOTransition,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # Rerun network
                    actor_policy = actor_apply_fn(actor_params, traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    # Calculate actor loss
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config.system.clip_eps,
                            1.0 + config.system.clip_eps,
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    # The seed will be used in the TanhTransformedDistribution:
                    entropy = actor_policy.entropy(seed=key).mean()

                    total_loss_actor = loss_actor - config.system.ent_coef * entropy
                    return total_loss_actor, (loss_actor, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict, traj_batch: PPOTransition, targets: chex.Array
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # Rerun network
                    value = critic_apply_fn(critic_params, traj_batch.obs)

                    # Calculate value loss
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    critic_total_loss = config.system.vf_coef * value_loss
                    return critic_total_loss, (value_loss)

                # Calculate actor loss
                key, entropy_key = jax.random.split(key)
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params, traj_batch, advantages, entropy_key
                )

                # Calculate critic loss
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                critic_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo notebook.
                # available at https://tinyurl.com/26tdzs5x
                # pmean over learner devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info),
                    axis_name="learner_devices",
                )

                # pmean over learner devices.
                critic_grads, critic_loss_info = jax.lax.pmean(
                    (critic_grads, critic_loss_info), axis_name="learner_devices"
                )

                # Update actor params and optimiser state
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                # Update critic params and optimiser state
                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                # Pack new params and optimiser state
                new_params = Params(actor_new_params, critic_new_params)
                new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)
                # Pack loss info
                actor_total_loss, (actor_loss, entropy) = actor_loss_info
                critic_total_loss, (value_loss) = critic_loss_info
                total_loss = critic_total_loss + actor_total_loss
                loss_info = {
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }
                return (new_params, new_opt_state, key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)
            # Shuffle minibatches
            batch_size = config.system.rollout_length * num_learner_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = tree.map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
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
        learner_state = LearnerState(params, opt_states, key, None, learner_state.timestep)
        metric = traj_batch.info
        return learner_state, (metric, loss_info)

    def learner_fn(
        learner_state: LearnerState, traj_batch: PPOTransition
    ) -> ExperimentOutput[LearnerState]:
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
        learner_state, (episode_info, loss_info) = _update_step(learner_state, traj_batch)

        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_thread(
    learn_fn: SebulbaLearnerFn[LearnerState, PPOTransition],
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
                # Accumulate the batches, timesteps, and rollout times
                accumulated_traj_batches = []
                accumulated_timesteps = []

                # Possibly get many rollouts for 1 learn step - allows learning with large batches
                for _ in range(config.arch.n_learner_accumulate):
                    # Get the trajectory batch from the pipeline
                    # This is blocking so it will wait until the pipeline has data.
                    with RecordTimeTo(learn_times["rollout_get_time"]):
                        traj_batch, timestep, rollout_time = pipeline.get(block=True)

                    # Store the retrieved data
                    accumulated_traj_batches.append(traj_batch)
                    accumulated_timesteps.append(timestep)
                    rollout_times.append(rollout_time)

                # Concatenate the accumulated timesteps and trajectory batches on the num_envs axis
                traj_batches = tree.map(lambda *x: jnp.concat(x, axis=0), *accumulated_traj_batches)
                timesteps = tree.map(lambda *x: jnp.concat(x, axis=0), *accumulated_timesteps)

                # Replace the timestep in the learner state with the latest timestep
                # This means the learner has access to the entire trajectory as well as
                # an additional timestep which it can use to bootstrap.
                learner_state = learner_state._replace(timestep=timesteps)
                # Update the networks
                with RecordTimeTo(learn_times["learning_time"]):
                    learner_state, ep_metrics, train_metrics = learn_fn(learner_state, traj_batches)

                metrics.append((ep_metrics, train_metrics))

                # Update all the params sources so all actors can get the latest params
                for source in params_sources:
                    source.update(learner_state.params)

        # Pass all the metrics and  params to the main thread (evaluator) for logging and evaluation
        ep_metrics, train_metrics = tree.map(lambda *x: np.asarray(x), *metrics)
        rollout_times: Dict[str, NDArray] = tree.map(lambda *x: np.mean(x), *rollout_times)
        timing_dict = rollout_times | learn_times
        timing_dict = tree.map(np.mean, timing_dict, is_leaf=lambda x: isinstance(x, list))

        eval_queue.put((ep_metrics, train_metrics, learner_state, timing_dict))


def learner_setup(
    key: chex.PRNGKey, config: DictConfig, learner_devices: List
) -> Tuple[
    SebulbaLearnerFn[LearnerState, PPOTransition],
    Tuple[ActorApply, CriticApply],
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
    model_spec = P()
    data_spec = P("learner_devices")
    learner_sharding = NamedSharding(mesh, model_spec)

    # PRNG keys.
    key, actor_key, critic_key = jax.random.split(key, 3)

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_action_head = hydra.utils.instantiate(
        config.network.action_head, action_dim=config.system.num_actions
    )
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = Critic(torso=critic_torso)

    actor_lr = make_learning_rate(config.system.actor_lr, config)
    critic_lr = make_learning_rate(config.system.critic_lr, config)

    actor_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_optim = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialise observation: Select only obs for a single agent.
    init_obs = jnp.array([env.single_observation_space.sample()])
    init_action_mask = jnp.ones((config.system.num_agents, config.system.num_actions))
    init_x = Observation(init_obs, init_action_mask)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_params = critic_network.init(critic_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = Params(actor_params, critic_params)

    # Pack apply and update functions.
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # defines how the learner state is sharded: params, opt and key = replicated, timestep = sharded
    learn_state_spec = LearnerState(model_spec, model_spec, model_spec, None, data_spec)
    learn = get_learner_step_fn(apply_fns, update_fns, config)
    learn = jax.jit(
        shard_map(
            learn,
            mesh=mesh,
            in_specs=(learn_state_spec, data_spec),
            out_specs=ExperimentOutput(learn_state_spec, data_spec, data_spec),
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
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)

    # Duplicate learner across Learner devices.
    params, opt_states, step_keys = jax.device_put(
        (params, opt_states, step_keys), learner_sharding
    )

    # Initialise learner state.
    init_learner_state = LearnerState(params, opt_states, step_keys, None, None)  # type: ignore
    env.close()

    return learn, apply_fns, init_learner_state, learner_sharding  # type: ignore


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
    learn, apply_fns, learner_state, learner_sharding = learner_setup(key, config, learner_devices)

    # Setup evaluator.
    # One key per device for evaluation.
    eval_act_fn = make_ff_eval_act_fn(apply_fns[0], config)
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
        * config.arch.n_learner_accumulate
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
    pipe = Pipeline(config.arch.rollout_queue_size, learner_sharding, pipe_lifetime)
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
                    apply_fns,
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
    best_params_cpu = jax.device_get(inital_params.actor_params)

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
        eval_metrics = evaluator(learner_state_cpu.params.actor_params, eval_key, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)

        episode_return = np.mean(eval_metrics["episode_return"])

        if save_checkpoint:  # Save a checkpoint of the learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=learner_state_cpu,
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params_cpu = copy.deepcopy(learner_state_cpu.params.actor_params)
            max_episode_return = episode_return

    evaluator_envs.close()
    eval_performance = float(np.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
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
    config_path="../../../configs/default/",
    config_name="ff_ippo_sebulba.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)
    cfg.logger.system_name = "ff_ippo_sebulba"

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}IPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
