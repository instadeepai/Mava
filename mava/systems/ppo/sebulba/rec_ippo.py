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
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jax import tree
from jax.experimental import mesh_utils
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, NamedSharding, PartitionSpec, Sharding
from numpy.typing import NDArray
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import get_num_eval_envs, make_rec_eval_act_fn
from mava.evaluator import get_sebulba_eval_fn as get_eval_fn
from mava.networks import RecurrentActor as Actor
from mava.networks import RecurrentValueNet as Critic
from mava.networks.base import ScannedRNN
from mava.systems.ppo.types import (
    HiddenStates,
    OptStates,
    Params,
    RNNLearnerState,
    RNNPPOTransition,
)
from mava.types import (
    Metrics,
    Observation,
    RecActorApply,
    RecCriticApply,
    SebulbaLearnerFn,
)
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.config import ppo_sebulba_checks as check_sebulba_config
from mava.utils.jax_utils import switch_leading_axes
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.sebulba.pipelines import Pipeline
from mava.utils.sebulba.utils import ParamsSource, RecordTimeTo, stop_sebulba
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics
from mava.wrappers.gym import GymToJumanji


def rollout(
    key: chex.PRNGKey,
    env: GymToJumanji,
    config: DictConfig,
    rollout_queue: Pipeline,
    params_source: ParamsSource,
    apply_fns: Tuple[RecActorApply, RecCriticApply],
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
        dones: chex.Array,
        hstates: HiddenStates,
        key: chex.PRNGKey,
    ) -> Tuple:
        """Get action and value."""

        batched_observation = tree.map(lambda x: x[jnp.newaxis, :], observation)
        ac_in = (batched_observation, dones[jnp.newaxis, :])
        policy_hidden_state, actor_policy = actor_apply_fn(
            params.actor_params, hstates.policy_hidden_state, ac_in
        )
        critic_hidden_state, value = critic_apply_fn(
            params.critic_params, hstates.critic_hidden_state, ac_in
        )

        action = actor_policy.sample(seed=key).squeeze(0)
        log_prob = actor_policy.log_prob(action).squeeze(0)
        value = value.squeeze(0)

        hstates = HiddenStates(policy_hidden_state, critic_hidden_state)
        return action, log_prob, value, hstates

    timestep = env.reset(seed=seeds)

    # Initialise hidden states.
    init_policy_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )
    init_critic_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )
    last_hstates = HiddenStates(init_policy_hstate, init_critic_hstate)
    last_hstates = move_to_device(last_hstates)

    # Loop till the desired num_updates is reached.
    while not stop_event.is_set():
        # Rollout
        traj: List[RNNPPOTransition] = []
        episode_metrics: List[Dict] = []
        actor_timings: Dict[str, List[float]] = defaultdict(list)
        with RecordTimeTo(actor_timings["rollout_time"]):
            for _ in range(config.system.rollout_length):
                with RecordTimeTo(actor_timings["get_params_time"]):
                    params = params_source.get()  # Get the latest parameters from the learner

                last_obs = move_to_device(timestep.observation)
                last_dones = np.repeat(timestep.last(), num_agents).reshape(num_envs, -1)
                last_dones = move_to_device(last_dones)

                # Sample action from the policy.
                with RecordTimeTo(actor_timings["compute_action_time"]):
                    key, act_key = jax.random.split(key)
                    action, log_prob, value, hstates = act_fn(
                        params, last_obs, last_dones, last_hstates, act_key
                    )
                    cpu_action = jax.device_get(action)

                # Step environment
                with RecordTimeTo(actor_timings["env_step_time"]):
                    timestep = env.step(cpu_action)

                # Append data to storage
                traj.append(
                    RNNPPOTransition(
                        last_dones,
                        action,
                        value,
                        timestep.reward,
                        log_prob,
                        last_obs,
                        last_hstates,
                    )
                )
                last_hstates = hstates
                episode_metrics.append(timestep.extras["episode_metrics"])

        # Send trajectories to learner
        with RecordTimeTo(actor_timings["rollout_put_time"]):
            try:
                rollout_queue.put(traj, (actor_timings, episode_metrics), (timestep, hstates))
            except queue.Full:
                err = "Waited too long to add to the rollout queue, killing the actor thread"
                warnings.warn(err, stacklevel=2)
                break

    env.close()


def get_learner_step_fn(
    apply_fns: Tuple[RecActorApply, RecCriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> SebulbaLearnerFn[RNNLearnerState, RNNPPOTransition]:
    """Get the learner function."""

    num_envs = config.arch.num_envs
    num_learner_envs = int(num_envs // len(config.arch.learner_device_ids))

    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state: RNNLearnerState,
        traj_batch: RNNPPOTransition,
    ) -> Tuple[RNNLearnerState, Metrics]:
        """A single update of the network.

        This function calculates advantages and targets based on the trajectories
        from the actor and updates the actor and critic networks based on the losses.

        Args:
            learner_state (LearnerState): contains all the items needed for learning.
            traj_batch (PPOTransition): the batch of data to learn with.
        """

        # Add a batch dimension to the observation.
        (params, opt_states, key, env_state, last_timestep, last_done, hstates) = learner_state

        batched_last_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
        ac_in = (batched_last_observation, last_done[jnp.newaxis, :])

        # Run the network.
        _, last_val = critic_apply_fn(params.critic_params, hstates.critic_hidden_state, ac_in)
        # Squeeze out the batch dimension and mask out the value of terminal states.
        last_val = last_val.squeeze(0)
        # Calculate advantage
        advantages, targets = calculate_gae(
            traj_batch, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple[Tuple, Metrics]:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                # Unpack train state and batch info
                params, opt_states, key = train_state
                traj_batch, advantages, targets = batch_info

                def _actor_loss_fn(
                    actor_params: FrozenDict,
                    traj_batch: RNNPPOTransition,
                    gae: chex.Array,
                    key: chex.PRNGKey,
                ) -> Tuple:
                    """Calculate the actor loss."""
                    # Rerun network

                    obs_and_done = (traj_batch.obs, traj_batch.done)
                    _, actor_policy = actor_apply_fn(
                        actor_params, traj_batch.hstates.policy_hidden_state[0], obs_and_done
                    )
                    log_prob = actor_policy.log_prob(traj_batch.action)

                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
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
                    # The seed will be used in the TanhTransformedDistribution:
                    entropy = actor_policy.entropy(seed=key).mean()

                    total_loss = actor_loss - config.system.ent_coef * entropy
                    return total_loss, (actor_loss, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: RNNPPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # Rerun network
                    obs_and_done = (traj_batch.obs, traj_batch.done)
                    _, value = critic_apply_fn(
                        critic_params, traj_batch.hstates.critic_hidden_state[0], obs_and_done
                    )

                    # Calculate value loss
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_loss = config.system.vf_coef * value_loss
                    return total_loss, value_loss

                # Calculate actor loss
                key, entropy_key = jax.random.split(key)
                actor_grad_fn = jax.value_and_grad(_actor_loss_fn, has_aux=True)
                actor_loss_info, actor_grads = actor_grad_fn(
                    params.actor_params, traj_batch, advantages, entropy_key
                )

                # Calculate critic loss
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                value_loss_info, critic_grads = critic_grad_fn(
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
                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="learner_devices"
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
                value_loss, (value_loss) = value_loss_info
                total_loss = value_loss + actor_total_loss
                loss_info = {
                    "total_loss": total_loss,
                    "value_loss": value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }
                return (new_params, new_opt_state, key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state

            key = jnp.squeeze(key, axis=0)  # Remove the learner_devices axis
            key, shuffle_key, entropy_key = jax.random.split(key, 3)
            key = jnp.expand_dims(key, axis=0)  # Add the learner_devices axis for shape consitency

            # Shuffle minibatches
            batch = (traj_batch, advantages, targets)

            num_recurrent_chunks = (
                config.system.rollout_length // config.system.recurrent_chunk_size
            )

            batch_size = num_learner_envs * num_recurrent_chunks
            batch = tree.map(
                lambda x: x.reshape(
                    config.system.recurrent_chunk_size,
                    batch_size,
                    *x.shape[2:],
                ),
                batch,
            )
            permutation = jax.random.permutation(
                shuffle_key, num_learner_envs * num_recurrent_chunks
            )
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=1), batch)
            reshaped_batch = tree.map(
                lambda x: jnp.reshape(
                    x, (x.shape[0], config.system.num_minibatches, -1, *x.shape[2:])
                ),
                shuffled_batch,
            )
            minibatches = tree.map(lambda x: jnp.swapaxes(x, 1, 0), reshaped_batch)

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

        # hstates is replaced in learner thread
        learner_state = RNNLearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            None,  # type: ignore
        )
        return learner_state, loss_info

    def learner_fn(
        learner_state: RNNLearnerState, traj_batch: RNNPPOTransition
    ) -> Tuple[RNNLearnerState, Metrics]:
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
                - dones (bool): Whether the initial timestep was a terminal state.
                - hstateS (HiddenStates): The initial hidden states of the RNN.
        """
        # This function is shard mapped on the batch axis, but `_update_step` needs
        # the first axis to be time
        traj_batch = tree.map(switch_leading_axes, traj_batch)
        learner_state, loss_info = _update_step(learner_state, traj_batch)

        return learner_state, loss_info

    return learner_fn


def learner_thread(
    learn_fn: SebulbaLearnerFn[RNNLearnerState, RNNPPOTransition],
    learner_state: RNNLearnerState,
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
                    traj_batch, rollout_time, ep_metrics, (timestep, hstates) = pipeline.get(  # type: ignore
                        block=True
                    )

                # Replace the timestep in the learner state with the latest timestep
                # This means the learner has access to the entire trajectory as well as
                # an additional timestep which it can use to bootstrap.
                learner_state = learner_state._replace(
                    timestep=timestep,
                    dones=timestep.last()
                    .repeat(config.system.num_agents)
                    .reshape(config.arch.num_envs, -1),
                    hstates=hstates,  # type: ignore
                )
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
    SebulbaLearnerFn[RNNLearnerState, RNNPPOTransition],
    Tuple[RecActorApply, RecCriticApply],
    RNNLearnerState,
    Sharding,
]:
    """Initialise learner_fn, network and learner state."""

    # Create temporory envoirnments.
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
    key, actor_key, critic_key = jax.random.split(key, 3)

    # Define network and optimisers.
    actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)

    action_head, _ = get_action_head(action_space)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=config.system.num_actions)

    critic_pre_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)
    critic_post_torso = hydra.utils.instantiate(config.network.critic_network.post_torso)

    actor_network = Actor(
        pre_torso=actor_pre_torso,
        post_torso=actor_post_torso,
        action_head=actor_action_head,
        hidden_state_dim=config.network.hidden_state_dim,
    )
    critic_network = Critic(
        pre_torso=critic_pre_torso,
        post_torso=critic_post_torso,
        hidden_state_dim=config.network.hidden_state_dim,
    )

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
    single_obs = jnp.array([[env.single_observation_space.sample()]])
    init_action_mask = jnp.ones((1, config.system.num_agents, config.system.num_actions))
    init_obs = Observation(single_obs, init_action_mask)
    init_done = jnp.zeros((1, config.arch.num_envs, config.system.num_agents), dtype=bool)
    init_x = (init_obs, init_done)

    # Initialise hidden states.
    init_policy_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, config.system.num_agents), config.network.hidden_state_dim
    )
    init_critic_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, config.system.num_agents), config.network.hidden_state_dim
    )

    # Initialise params and optimiser state.
    actor_params = actor_network.init(actor_key, init_policy_hstate, init_x)
    actor_opt_state = actor_optim.init(actor_params)
    critic_params = critic_network.init(critic_key, init_critic_hstate, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params and initial states.
    params = Params(actor_params, critic_params)
    hstates = HiddenStates(init_policy_hstate, init_critic_hstate)

    # Pack apply and update functions.
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # Defines how the learner state is sharded: params, opt and key = replicated, timestep = sharded
    learn_state_spec = RNNLearnerState(
        model_spec, model_spec, data_spec, None, data_spec, data_spec, data_spec
    )

    learn = get_learner_step_fn(apply_fns, update_fns, config)
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

    opt_states = OptStates(actor_opt_state, critic_opt_state)
    dones = jnp.zeros(
        (config.arch.num_envs, config.system.num_agents),
        dtype=bool,
    )

    # Duplicate learner across Learner devices.
    params, opt_states, hstates, step_keys, dones = jax.device_put(
        (params, opt_states, hstates, step_keys, dones), learner_sharding
    )

    # Initialise learner state.
    init_learner_state = RNNLearnerState(params, opt_states, step_keys, None, None, dones, None)  # type: ignore
    env.close()

    return learn, apply_fns, init_learner_state, learner_sharding


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "rec_ippo_sebulba"
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

    # Set recurrent chunk size.
    if config.system.recurrent_chunk_size is None:
        config.system.recurrent_chunk_size = config.system.rollout_length
    else:
        assert (
            config.system.rollout_length % config.system.recurrent_chunk_size == 0
        ), "Rollout length must be divisible by recurrent chunk size."

    # Setup learner.
    learn, apply_fns, learner_state, learner_sharding = learner_setup(key, config, learner_devices)

    # Setup evaluator.
    # One key per device for evaluation.
    eval_act_fn = make_rec_eval_act_fn(apply_fns[0], config)
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

    # Create an initial hidden state used for resetting memory for evaluation
    eval_batch_size = min(config.arch.num_eval_episodes, config.arch.num_envs)
    eval_hs = ScannedRNN.initialize_carry(
        (eval_batch_size, config.system.num_agents),
        config.network.hidden_state_dim,
    )

    # Executor setup and launch.
    inital_params = jax.device_put(learner_state.params, actor_devices[0])  # unreplicate

    # The rollout queue/ the pipe between actor and learner
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
                    apply_fns,
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
        eval_metrics = evaluator(
            learner_state_cpu.params.actor_params, eval_key, {"hidden_state": eval_hs}
        )
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
            max_episode_return = float(episode_return)

    evaluator_envs.close()
    eval_performance = float(np.mean(eval_metrics[config.env.eval_metric]))

    # Gracefully shutting down all actors and resources.
    stop_sebulba(actors_stop_event, pipe, params_sources, actor_threads)

    # Measure absolute metric.
    if config.arch.absolute_metric:
        print(f"{Fore.BLUE}{Style.BRIGHT}Measuring absolute metric...{Style.RESET_ALL}")

        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        eval_hs = ScannedRNN.initialize_carry(
            (eval_batch_size, config.system.num_agents),
            config.network.hidden_state_dim,
        )

        abs_metric_evaluator, abs_metric_evaluator_envs = get_eval_fn(
            environments.make_gym_env, eval_act_fn, config, np_rng, absolute_metric=True
        )
        key, eval_key = jax.random.split(key, 2)
        eval_metrics = abs_metric_evaluator(best_params_cpu, eval_key, {"hidden_state": eval_hs})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)
        abs_metric_evaluator_envs.close()

    # Stop all the threads.
    logger.stop()

    return eval_performance


@hydra.main(
    config_path="../../../configs/default/",
    config_name="rec_ippo_sebulba.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Recurrent IPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
