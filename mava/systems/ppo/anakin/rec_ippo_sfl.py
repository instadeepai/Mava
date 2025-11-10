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
from functools import partial
from typing import Any, Callable, Tuple

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import jumanji
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jax import tree
from jumanji.environments.routing.connector.generator import (
    RandomWalkGenerator,
    UniformRandomGenerator,
)
from jumanji.environments.routing.connector.generator.random_walk_generator import (
    StochasticRandomWalkGenerator,
)
from jumanji.environments.routing.connector.reward import (
    DenseRewardFn,
    RewardFn,
    SharedDenseRewardFn,
    SharedSparseRewardFn,
    SparseRewardFn,
)
from jumanji.environments.routing.connector.types import State
from omegaconf import DictConfig, OmegaConf

from mava.evaluator import (
    get_eval_fn,
    get_num_eval_envs,
    get_singleton_eval_fn,
    make_rec_eval_act_fn,
)
from mava.networks import RecurrentActor as Actor
from mava.networks import RecurrentValueNet as Critic
from mava.networks import ScannedRNN
from mava.systems.ppo.types import (
    HiddenStates,
    OptStates,
    Params,
    RNNLearnerState,
    RNNPPOTransition,
)
from mava.types import (
    ExperimentOutput,
    LearnerFn,
    MarlEnv,
    Metrics,
    RecActorApply,
    RecCriticApply,
)
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.connector_eval import get_eval_envs
from mava.utils.jax_utils import unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.make_env import _jumanji_registry
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.auto_reset_wrapper import DeterministicAutoResetWrapper
from mava.wrappers.episode_metrics import RecordEpisodeMetrics, get_final_step_metrics
from mava.wrappers.jumanji import VectorConnectorWrapper


class CompleteSparseRewardFn(RewardFn):
    def __call__(self, state: State, action: chex.Array, next_state: State) -> float:
        all_connected = jnp.all(next_state.agents.connected) & ~jnp.all(state.agents.connected)
        num_agents = state.agents.id.shape[0]
        return all_connected.repeat(num_agents) * 1.0


def make_jumanji_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    # Config generator and select the wrapper.
    if config.env.generator == "random_walk":
        train_generator = RandomWalkGenerator(**config.env.scenario.task_config)
    elif config.env.generator == "uniform":
        train_generator = UniformRandomGenerator(**config.env.scenario.task_config)
    elif config.env.generator == "stochastic":
        train_generator = StochasticRandomWalkGenerator(**config.env.scenario.task_config)
    else:
        raise ValueError(f"Generator {config.env.generator} not supported.")

    if config.env.reward_fn == "shared_dense":
        reward_fn = SharedDenseRewardFn()
    elif config.env.reward_fn == "dense":
        reward_fn = DenseRewardFn()
    elif config.env.reward_fn == "shared_sparse":
        reward_fn = SharedSparseRewardFn()
    elif config.env.reward_fn == "sparse":
        reward_fn = SparseRewardFn()
    elif config.env.reward_fn == "complete_sparse":
        reward_fn = CompleteSparseRewardFn()
    else:
        raise ValueError(f"Reward function {config.env.reward_fn} not supported.")

    eval_generator = RandomWalkGenerator(**config.env.scenario.task_config)
    wrapper = _jumanji_registry[config.env.env_name]["wrapper"]

    # Create envs.
    env_config = {**config.env.kwargs, **config.env.scenario.env_kwargs}
    train_env = jumanji.make(
        config.env.scenario.name, generator=train_generator, reward_fn=reward_fn, **env_config
    )
    eval_env = jumanji.make(
        config.env.scenario.name, generator=eval_generator, reward_fn=reward_fn, **env_config
    )
    train_env = wrapper(train_env, add_global_state=add_global_state)
    eval_env = wrapper(eval_env, add_global_state=add_global_state)
    return train_env, eval_env


def make_env(config: DictConfig) -> MarlEnv:
    train_env, eval_env = make_jumanji_env(config)
    eval_envs = get_eval_envs()

    # Disable the AgentID wrapper if the environment has implicit agent IDs.
    # config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)

    # if config.system.add_agent_id:
    #     train_env = AgentIDWrapper(train_env)
    #     eval_env = AgentIDWrapper(eval_env)
    #     eval_envs = [AgentIDWrapper(e) for e in eval_envs]

    train_env = DeterministicAutoResetWrapper(train_env)
    eval_envs = [RecordEpisodeMetrics(VectorConnectorWrapper(e)) for e in eval_envs]
    train_env = RecordEpisodeMetrics(train_env)
    eval_env = RecordEpisodeMetrics(eval_env)
    # eval_envs = [RecordEpisodeMetrics(e) for e in eval_envs]

    return train_env, eval_env, eval_envs


def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple[RecActorApply, RecCriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[RNNLearnerState]:
    """Get the learner function."""
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(
        learner_state_learnable_env_state: Tuple[RNNLearnerState, Any], _: Any
    ) -> Tuple[RNNLearnerState, Tuple]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The current model parameters.
                - opt_states (OptStates): The current optimizer states.
                - key (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
                - dones (bool): Whether the last timestep was a terminal state.
                - hstates (HiddenStates): The current hidden states of the RNN.
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state_with_start_state: Tuple[RNNLearnerState, Any], _: Any
        ) -> Tuple[RNNLearnerState, Tuple[RNNPPOTransition, Metrics]]:
            """Step the environment."""
            learner_state, start_state = learner_state_with_start_state
            (
                params,
                opt_states,
                key,
                env_state,
                last_timestep,
                last_done,
                last_hstates,
            ) = learner_state

            key, policy_key = jax.random.split(key)

            # Add a batch dimension to the observation.
            batched_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
            ac_in = (batched_observation, last_done[jnp.newaxis, :])

            # Run the network.
            policy_hidden_state, actor_policy = actor_apply_fn(
                params.actor_params, last_hstates.policy_hidden_state, ac_in
            )
            critic_hidden_state, value = critic_apply_fn(
                params.critic_params, last_hstates.critic_hidden_state, ac_in
            )

            # Sample action from the policy and squeeze out the batch dimension.
            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)
            value, action, log_prob = value.squeeze(0), action.squeeze(0), log_prob.squeeze(0)

            # Step the environment.
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0, 0))(
                env_state, action, start_state
            )

            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)
            hstates = HiddenStates(policy_hidden_state, critic_hidden_state)
            transition = RNNPPOTransition(
                last_done,
                action,
                value,
                timestep.reward,
                log_prob,
                last_timestep.observation,
                last_hstates,
            )
            learner_state = RNNLearnerState(
                params, opt_states, key, env_state, timestep, done, hstates
            )
            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]
            learner_state_with_start_state = (learner_state, start_state)
            return learner_state_with_start_state, (transition, metrics)

        # jax.debug.print("Start of learn")
        # Sample learnable states and random states
        # TODO: fix this so that it doesn't always reset the environment
        learner_state, learnable_env_state = learner_state_learnable_env_state

        key, sampled_key, gen_key = jax.random.split(learner_state.key, 3)

        # Sample learnable states
        sampled_key_0, sampled_key_1 = jax.random.split(sampled_key, 2)
        sampled_idxs = jax.random.randint(
            sampled_key_0,
            (config.ued.num_sampled,),
            0,
            config.ued.num_to_save,
        )
        env_state_sampled = jax.tree_util.tree_map(lambda x: x[sampled_idxs], learnable_env_state)

        # Generate random states
        gen_keys = jax.random.split(gen_key, config.arch.num_envs - config.ued.num_sampled)
        env_state_gen, _ = jax.vmap(env.reset)(gen_keys)

        # Concatenate sampled and generated states
        state_re = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            env_state_gen,
            env_state_sampled,
        )
        learner_state_with_reset_state = (learner_state, state_re)

        # jax.debug.print("Start gettign traj")
        # Step environment for rollout length
        learner_state_with_reset_state, (traj_batch, episode_metrics) = jax.lax.scan(
            _env_step, learner_state_with_reset_state, None, config.system.rollout_length
        )
        learner_state, _ = learner_state_with_reset_state

        # jax.debug.print("Start updating")
        # Calculate advantage
        params, opt_states, key, env_state, last_timestep, last_done, hstates = learner_state

        # Add a batch dimension to the observation.
        batched_last_observation = tree.map(lambda x: x[jnp.newaxis, :], last_timestep.observation)
        ac_in = (batched_last_observation, last_done[jnp.newaxis, :])

        # Run the network.
        _, last_val = critic_apply_fn(params.critic_params, hstates.critic_hidden_state, ac_in)
        # Squeeze out the batch dimension and mask out the value of terminal states.
        last_val = last_val.squeeze(0)

        advantages, targets = calculate_gae(
            traj_batch, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
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

                    # Clipped MSE loss
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
                    params.actor_params,
                    traj_batch,
                    advantages,
                    entropy_key,
                )

                # Calculate critic loss
                critic_grad_fn = jax.value_and_grad(_critic_loss_fn, has_aux=True)
                value_loss_info, critic_grads = critic_grad_fn(
                    params.critic_params, traj_batch, targets
                )

                # Compute the parallel mean (pmean) over the batch.
                # This pmean could be a regular mean as the batch axis is on the same device.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="batch"
                )
                # pmean over devices.
                actor_grads, actor_loss_info = jax.lax.pmean(
                    (actor_grads, actor_loss_info), axis_name="device"
                )

                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="batch"
                )
                # pmean over devices.
                critic_grads, value_loss_info = jax.lax.pmean(
                    (critic_grads, value_loss_info), axis_name="device"
                )

                # Update params and optimiser state
                actor_updates, actor_new_opt_state = actor_update_fn(
                    actor_grads, opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

                critic_updates, critic_new_opt_state = critic_update_fn(
                    critic_grads, opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

                new_params = Params(actor_new_params, critic_new_params)
                new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)

                actor_loss, (_, entropy) = actor_loss_info
                value_loss, unscaled_value_loss = value_loss_info

                total_loss = actor_loss + value_loss
                loss_info = {
                    "total_loss": total_loss,
                    "value_loss": unscaled_value_loss,
                    "actor_loss": actor_loss,
                    "entropy": entropy,
                }

                return (new_params, new_opt_state, entropy_key), loss_info

            params, opt_states, traj_batch, advantages, targets, key = update_state
            key, shuffle_key, entropy_key = jax.random.split(key, 3)

            # Shuffle minibatches
            batch = (traj_batch, advantages, targets)
            num_recurrent_chunks = (
                config.system.rollout_length // config.system.recurrent_chunk_size
            )
            batch = tree.map(
                lambda x: x.reshape(
                    config.system.recurrent_chunk_size,
                    config.arch.num_envs * num_recurrent_chunks,
                    *x.shape[2:],
                ),
                batch,
            )
            permutation = jax.random.permutation(
                shuffle_key, config.arch.num_envs * num_recurrent_chunks
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
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
                _update_minibatch, (params, opt_states, entropy_key), minibatches
            )

            update_state = (
                params,
                opt_states,
                traj_batch,
                advantages,
                targets,
                key,
            )
            return update_state, loss_info

        update_state = (
            params,
            opt_states,
            traj_batch,
            advantages,
            targets,
            key,
        )

        # Update epochs
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config.system.ppo_epochs
        )

        params, opt_states, traj_batch, advantages, targets, key = update_state
        learner_state = RNNLearnerState(
            params,
            opt_states,
            key,
            env_state,
            last_timestep,
            last_done,
            hstates,
        )
        learner_state_learnable_env_state = (learner_state, learnable_env_state)
        return learner_state_learnable_env_state, (episode_metrics, loss_info)

    def learner_fn(
        learner_state_with_learnable_instances: Tuple[RNNLearnerState, Any],
    ) -> ExperimentOutput[RNNLearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer states.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
                - dones (bool): Whether the initial timestep was a terminal state.
                - hstateS (HiddenStates): The initial hidden states of the RNN.

        """
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state_with_learnable_instances, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step,
            learner_state_with_learnable_instances,
            None,
            config.system.num_updates_per_eval,
        )
        return ExperimentOutput(
            learner_state=learner_state_with_learnable_instances[0],
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[RNNLearnerState], Actor, RNNLearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    num_agents = env.num_agents
    config.system.num_agents = num_agents

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimisers.
    actor_pre_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    actor_post_torso = hydra.utils.instantiate(config.network.actor_network.post_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
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

    # Initialise observation with obs of all agents.
    init_obs = env.observation_spec.generate_value()
    init_obs = tree.map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config.arch.num_envs, axis=0),
        init_obs,
    )
    init_obs = tree.map(lambda x: x[jnp.newaxis, ...], init_obs)
    init_done = jnp.zeros((1, config.arch.num_envs, num_agents), dtype=bool)
    init_x = (init_obs, init_done)

    # Initialise hidden states.
    init_policy_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )
    init_critic_hstate = ScannedRNN.initialize_carry(
        (config.arch.num_envs, num_agents), config.network.hidden_state_dim
    )

    # initialise params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_policy_hstate, init_x)
    actor_opt_state = actor_optim.init(actor_params)
    critic_params = critic_network.init(critic_net_key, init_critic_hstate, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Get network apply functions and optimiser updates.
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

    # Pack params and initial states.
    params = Params(actor_params, critic_params)
    hstates = HiddenStates(init_policy_hstate, init_critic_hstate)

    # Load model from checkpoint if specified.
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,  # Other checkpoint args
        )
        # Restore the learner state from the checkpoint
        restored_params, restored_hstates = loaded_checkpoint.restore_params(
            input_params=params, restore_hstates=True, THiddenState=HiddenStates
        )
        # Update the params and hstates
        params = restored_params
        hstates = restored_hstates if restored_hstates else hstates

    # Initialise environment states and timesteps: across devices and batches.
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_keys),
    )
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    # (devices, update batch size, num_envs, ...)
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Define params to be replicated across devices and batches.
    dones = jnp.zeros(
        (config.arch.num_envs, num_agents),
        dtype=bool,
    )
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, hstates, step_keys, dones)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, hstates, step_keys, dones = replicate_learner
    init_learner_state = RNNLearnerState(
        params=params,
        opt_states=opt_states,
        key=step_keys,
        env_state=env_states,
        timestep=timesteps,
        dones=dones,
        hstates=hstates,
    )
    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "rec_ippo"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Set recurrent chunk size.
    if config.system.recurrent_chunk_size is None:
        config.system.recurrent_chunk_size = config.system.rollout_length
    else:
        assert (
            config.system.rollout_length % config.system.recurrent_chunk_size == 0
        ), "Rollout length must be divisible by recurrent chunk size."

        assert (
            config.arch.num_envs % config.system.num_minibatches == 0
        ), "Number of envs must be divisibile by number of minibatches."

    # Create the enviroments for train and eval.
    env, eval_env, eval_envs = make_env(config)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    # Setup evaluator.
    # One key per device for evaluation.
    eval_keys = jax.random.split(key_e, n_devices)
    eval_act_fn = make_rec_eval_act_fn(actor_network.apply, config)
    id_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)
    ood_evaluator = get_singleton_eval_fn(eval_envs, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation

    config.ued.num_sampled = int(config.ued.sampling_ratio * config.arch.num_envs)

    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )

    # Logger setup
    logger = MavaLogger(config)
    logger.log_config(OmegaConf.to_container(config, resolve=True))

    # Set up checkpointer
    save_checkpoint = config.logger.checkpointing.save_model
    if save_checkpoint:
        checkpointer = Checkpointer(
            metadata=config,  # Save all config as metadata in the checkpoint
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,  # Checkpoint args
        )

    # Create an initial hidden state used for resetting memory for evaluation
    eval_batch_size = get_num_eval_envs(config, absolute_metric=False)
    eval_hs = ScannedRNN.initialize_carry(
        (n_devices, eval_batch_size, config.system.num_agents),
        config.network.hidden_state_dim,
    )

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        key, learnable_key = jax.random.split(key)
        # print("Getting Learnable Instances")
        success_scores, learnability_scores, learnable_instances = get_learnability_set(
            learnable_key,
            unreplicate_n_dims(learner_state.params.actor_params),
            actor_network.apply,
            config,
            env,
        )
        # print("Finished Getting Learnable Instances")
        broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
        replicate_learnable_instances = tree.map(broadcast, learnable_instances)

        # Duplicate learnable states across devices.
        replicate_learnable_instances = flax.jax_utils.replicate(
            replicate_learnable_instances, devices=jax.devices()
        )
        learner_state_with_learnable_instances = (learner_state, replicate_learnable_instances)
        learner_output = learn(learner_state_with_learnable_instances)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        num_updates = (eval_step + 1) * config.system.num_updates_per_eval
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, num_updates, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, num_updates, eval_step, LogEvent.ACT)
        train_metrics = learner_output.train_metrics
        train_metrics["learnability"] = learnability_scores
        train_metrics["learnability_win_rate"] = success_scores
        logger.log(train_metrics, num_updates, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        # Evaluate.
        eval_metrics = id_evaluator(trained_params, eval_keys, {"hidden_state": eval_hs})
        # ood_eval_metrics = ood_evaluator(trained_params, eval_keys)
        # eval_metrics_log = {"id": eval_metrics, "ood": ood_eval_metrics}
        eval_metrics_log = {"id": eval_metrics}
        logger.log(
            eval_metrics, eval_step * config.system.num_updates_per_eval, eval_step, LogEvent.EVAL
        )
        episode_return = jnp.mean(eval_metrics["episode_return"])

        if save_checkpoint:
            # Save checkpoint of learner state
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update runner state to continue training.
        learner_state = learner_output.learner_state

    # Record the performance for the final evaluation run.
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric.
    if config.arch.absolute_metric:
        eval_batch_size = get_num_eval_envs(config, absolute_metric=True)
        eval_hs = ScannedRNN.initialize_carry(
            (n_devices, eval_batch_size, config.system.num_agents),
            config.network.hidden_state_dim,
        )
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {"hidden_state": eval_hs})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return eval_performance


def rollout_env_step_fn(
    rng: chex.PRNGKey,
    env_state: chex.Array,
    obs: chex.Array,
    last_done: chex.Array,
    last_hstate: chex.Array,
    actor_apply_fn: Callable,
    actor_params: FrozenDict,
    env: Any,
    reset_state: chex.Array,
) -> Tuple[chex.Array, chex.Array]:
    num_agents = last_done.shape[1]
    num_envs = last_done.shape[0]

    # Add a batch dimension to the observation.
    batched_observation = tree.map(lambda x: x[jnp.newaxis, :], obs)
    ac_in = (batched_observation, last_done[jnp.newaxis, :])

    # SELECT ACTION
    rng, policy_rng = jax.random.split(rng)
    hstate, actor_policy = actor_apply_fn(actor_params, last_hstate, ac_in)
    action = actor_policy.sample(seed=policy_rng)
    action = action.squeeze(0)

    # STEP ENVIRONMENT
    env_state, timestep = jax.vmap(env.step)(env_state, action, reset_state)
    # jax.lax.cond(timestep.extras['env_metrics']['Success'].sum() > 0, lambda: jax.debug.breakpoint(), lambda: None)

    # LOG EPISODE METRICS
    done = jnp.repeat(timestep.last(), num_agents)
    done = done.reshape(num_envs, -1)

    goal_reached = env_state.env_state.agents.connected

    metrics = (goal_reached, timestep.extras["env_metrics"]["won_episode"])

    return rng, env_state, timestep, done, hstate, metrics


@partial(jax.vmap, in_axes=(None, None, 1, 1))
@partial(jax.jit, static_argnums=(0, 1))
def calc_outcomes_by_agent(max_steps: int, max_episodes: int, dones, goal_reached):
    idxs = jnp.arange(max_steps)

    @partial(jax.vmap, in_axes=(0, 0))
    def _ep_outcomes(start_idx, end_idx):
        mask = (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
        success = jnp.max(goal_reached * mask)
        # jax.debug.breakpoint()
        return success

    done_idxs = jnp.argwhere(dones, size=max_episodes, fill_value=max_steps).squeeze()
    mask_done = jnp.where(done_idxs == max_steps, 0, 1)
    success = _ep_outcomes(jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs)

    # jax.debug.breakpoint()
    return {
        "success_rate": success.mean(where=mask_done),
    }


def test_calc_outcomes_by_agent():
    # 3 env, 2 agents, 10 steps
    max_steps = 10
    # dones: steps x envs x agents
    dones_e0_a0 = jnp.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1]).reshape(-1, 1, 1)
    dones_e0_a1 = jnp.array([0, 1, 0, 1, 0, 1, 0, 0, 0, 1]).reshape(-1, 1, 1)
    dones_e1_a0 = jnp.array([0, 0, 1, 0, 0, 0, 1, 0, 0, 1]).reshape(-1, 1, 1)
    dones_e1_a1 = jnp.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1]).reshape(-1, 1, 1)
    dones_e2_a0 = jnp.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1]).reshape(-1, 1, 1)
    dones_e2_a1 = jnp.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1]).reshape(-1, 1, 1)

    dones_a0 = jnp.concatenate([dones_e0_a0, dones_e1_a0, dones_e2_a0], axis=1)
    dones_a1 = jnp.concatenate([dones_e0_a1, dones_e1_a1, dones_e2_a1], axis=1)

    dones = jnp.concatenate([dones_a0, dones_a1], axis=2)
    goal_reached_e0_a0 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1, 1)
    goal_reached_e0_a1 = jnp.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1, 1)
    goal_reached_e1_a0 = jnp.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1, 1)
    goal_reached_e1_a1 = jnp.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 1]).reshape(-1, 1, 1)
    goal_reached_e2_a0 = jnp.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1]).reshape(-1, 1, 1)
    goal_reached_e2_a1 = jnp.array([0, 0, 1, 0, 0, 1, 0, 0, 0, 1]).reshape(-1, 1, 1)

    goal_reached_a0 = jnp.concatenate(
        [goal_reached_e0_a0, goal_reached_e1_a0, goal_reached_e2_a0], axis=1
    )
    goal_reached_a1 = jnp.concatenate(
        [goal_reached_e0_a1, goal_reached_e1_a1, goal_reached_e2_a1], axis=1
    )
    goal_reached = jnp.concatenate([goal_reached_a0, goal_reached_a1], axis=2)

    dones_by_agent = dones.reshape(max_steps, -1)
    goal_reached_by_agent = goal_reached.reshape(max_steps, -1)

    o = calc_outcomes_by_agent(max_steps, dones_by_agent, goal_reached_by_agent)

    success_by_env_current = o["success_rate"].reshape(2, 3)

    success_by_env_new = o["success_rate"].reshape(3, 2)

    print(o)


def get_learnability_set(
    rng, actor_params, actor_apply_fn, config, env: Any
) -> Tuple[chex.Array, chex.Array, Any]:
    def _batch_step(_, rng):
        def _env_step(runner_state, _: Any):
            """Step the environment."""
            rng, env_state, obs, last_done, last_hstate, reset_key = runner_state

            rng, env_state, timestep, done, hstate, metrics = rollout_env_step_fn(
                rng,
                env_state,
                obs,
                last_done,
                last_hstate,
                actor_apply_fn,
                actor_params,
                env,
                reset_key,
            )
            runner_state = (rng, env_state, timestep.observation, done, hstate, reset_key)
            return runner_state, (done, *metrics)

        # sample envs
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.ued.batch_size)
        env_state, timestep = jax.vmap(env.reset)(reset_rng)
        dones = jnp.zeros(
            (config.ued.batch_size, config.system.num_agents),
            dtype=bool,
        )
        hstate = ScannedRNN.initialize_carry(
            (config.ued.batch_size, config.system.num_agents), config.network.hidden_state_dim
        )
        runner_state = (rng, env_state, timestep.observation, dones, hstate, env_state)
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config.ued.rollout_steps
        )
        # print("traj batch done", traj_batch[0].shape)
        # print("traj batch gr", traj_batch[1].shape)
        dones_by_agent = traj_batch[0].reshape(config.ued.rollout_steps, -1)
        goal_reached_by_agent = traj_batch[1].reshape(config.ued.rollout_steps, -1)

        o = calc_outcomes_by_agent(
            config.ued.rollout_steps,
            config.ued.max_episodes,
            dones_by_agent,
            goal_reached_by_agent,
        )

        won_episode_outcomes = calc_outcomes_by_agent(
            config.ued.rollout_steps,
            config.ued.max_episodes,
            traj_batch[0][:, :, 0],
            traj_batch[2],
        )
        # jax.debug.print("won_episode_outcomes: {won_episode_outcomes}", won_episode_outcomes)
        success_by_env_0 = won_episode_outcomes["success_rate"]
        learnability_by_env_0 = success_by_env_0 * (1 - success_by_env_0)
        success_by_env = o["success_rate"].reshape(
            (config.ued.batch_size, config.system.num_agents)
        )
        learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=1)
        perfect_regret = 1 - success_by_env_0
        # print("learnability_by_env", learnability_by_env)
        # jax.debug.breakpoint()
        return None, (success_by_env_0, learnability_by_env_0, env_state)

    print("Starting get_learnability_set")

    rngs = jax.random.split(rng, config.ued.num_batches)
    _, (success, learnability, env_state) = jax.lax.scan(
        _batch_step, None, rngs, config.ued.num_batches
    )

    flat_env_state = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[2:]), env_state)
    learnability = learnability.flatten()
    flat_success = success.reshape((-1,) + success.shape[2:])
    top_k = jnp.argsort(learnability)[-config.ued.num_to_save :]
    # print("top 1000", top_1000)

    top_k_states = jax.tree.map(lambda x: x.at[top_k].get(), flat_env_state)
    # print("top 1000 instances", top_1000_env_state_ts)
    print("Finished get_learnability_set")
    return flat_success.at[top_k].get(), learnability.at[top_k].get(), top_k_states


def test_get_learnability_set(_config: DictConfig) -> None:
    """Tests get_learnability_set."""
    _config.logger.system_name = "rec_ippo_sfl"
    config = copy.deepcopy(_config)

    # Create the enviroments for train and eval.
    env, _ = make_env(config)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    single_actor_params = unreplicate_n_dims(learner_state.params.actor_params)
    success, learnability, top_instances = get_learnability_set(
        key, single_actor_params, actor_network.apply, config, env
    )

    # Validate the top instances by rolling them out
    print(f"\n{Fore.CYAN}{'=' * 80}")
    print(f"Validating Top {config.ued.num_to_save} Instances")
    print(f"{'=' * 80}{Style.RESET_ALL}\n")

    key, key_instance = jax.random.split(key_e)
    instance_keys = jax.random.split(key_instance, config.ued.num_to_save)
    env_state, timestep = jax.vmap(env.set_env_instance, in_axes=(0, 0))(
        top_instances, instance_keys
    )

    # Initialize hidden state for the actor
    dones = jnp.zeros((config.ued.num_to_save, env.num_agents), dtype=bool)
    hstate = ScannedRNN.initialize_carry(
        (config.ued.num_to_save, env.num_agents), config.network.hidden_state_dim
    )

    # Rollout function
    def _step(carry, _):
        rng, env_state, obs, last_done, last_hstate, start_state = carry

        rng, env_state, timestep, done, hstate, metrics = rollout_env_step_fn(
            rng,
            env_state,
            obs,
            last_done,
            last_hstate,
            actor_network.apply,
            single_actor_params,
            env,
            start_state,
        )
        new_carry = (rng, env_state, timestep.observation, done, hstate, start_state)

        return new_carry, (done, metrics[0], timestep.extras["env_metrics"]["win_rate"])

    # Need to fix this
    def _calc_success_rate(dones: chex.Array, successes: chex.Array) -> chex.Array:
        done_idxs = jnp.argwhere(dones.flatten()).squeeze()
        return successes.flatten().at[done_idxs].get().mean()

    # Run rollout
    start_state = env_state
    key, step_key = jax.random.split(key)
    initial_carry = (step_key, env_state, timestep.observation, dones, hstate, start_state)
    _, (dones_traj, goals_traj, successes_traj) = jax.lax.scan(
        _step, initial_carry, None, config.ued.rollout_steps
    )

    # Print results
    print(f"{Fore.GREEN}Validation Results:{Style.RESET_ALL}\n")
    print(f"{'Index':<8} {'Learnability':<15} {'Actual Success':<18} {'Per-Agent Success':<25}")
    print("-" * 80)

    for i in range(config.ued.num_to_save):
        learnability_score = float(learnability[i])

        o = calc_outcomes_by_agent(
            config.ued.rollout_steps, dones_traj[:, i, :], goals_traj[:, i, :]
        )["success_rate"]

        actual_success = _calc_success_rate(dones_traj[:, i, :], successes_traj[:, i, :])

        print(learnability_score)
        print(o.mean())
        print(actual_success)

    # Summary statistics
    avg_learnability = float(jnp.mean(learnability))
    avg_success = float(jnp.mean(successes_traj))

    print("\n" + "-" * 80)
    print(f"{Fore.CYAN}Summary:{Style.RESET_ALL}")
    print(f"  Average Learnability Score: {avg_learnability:.4f}")
    print(f"  Average Actual Success Rate: {avg_success:.4f}")
    print(
        f"  Target Range (0.2-0.8): {Fore.GREEN}✓{Style.RESET_ALL}"
        if 0.2 <= avg_success <= 0.8
        else f"{Fore.RED}✗{Style.RESET_ALL}"
    )
    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")


@hydra.main(
    config_path="../../../configs/default",
    config_name="rec_ippo_sfl.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    # test_calc_outcomes_by_agent()
    # _ = test_get_learnability_set(cfg)
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}Recurrent SFL IPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
