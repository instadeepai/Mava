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
from typing import Any, Tuple
from collections import namedtuple
from functools import partial

import chex
import flax
import hydra
import jax
import jax.numpy as jnp
import optax
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from jax import tree
from omegaconf import DictConfig, OmegaConf
import jaxmarl
from jaxmarl.environments.jaxnav.jaxnav_env import EnvInstance

from mava.evaluator import get_eval_fn, make_ff_eval_act_fn
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardValueNet as Critic
from mava.systems.ppo.types import LearnerState, OptStates, Params, PPOTransition
from mava.types import ActorApply, CriticApply, ExperimentOutput, LearnerFn, MarlEnv, Metrics
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_total_timesteps
from mava.utils.jax_utils import merge_leading_dims, unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics
from mava.wrappers.jaxmarl import JaxMarlWrapper, JaxNavWrapper
from mava.wrappers.observation import AgentIDWrapper
from mava.wrappers.episode_metrics import RecordEpisodeMetrics


SFLTransition = namedtuple("SFLTransition", ["done", "goal_reached"])

LearnerStateWithStartState = namedtuple("LearnerStateWithStartState", ["learner_state", "start_state"])

def make_env(config: DictConfig) -> MarlEnv:
    kwargs = dict(config.env.kwargs)
    # Create jaxmarl envs.
    train_env: MarlEnv = JaxNavWrapper(
        jaxmarl.make(config.env.scenario.name, **kwargs),
    )
    eval_env: MarlEnv = JaxNavWrapper(
        jaxmarl.make(config.env.scenario.name, **kwargs),
    )

    # Disable the AgentID wrapper if the environment has implicit agent IDs.
    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)

    if config.system.add_agent_id:
        train_env = AgentIDWrapper(train_env)
        eval_env = AgentIDWrapper(eval_env)

    train_env = RecordEpisodeMetrics(train_env)
    eval_env = RecordEpisodeMetrics(eval_env)

    return train_env, eval_env

def get_learner_fn(
    env: MarlEnv,
    apply_fns: Tuple[ActorApply, CriticApply],
    update_fns: Tuple[optax.TransformUpdateFn, optax.TransformUpdateFn],
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""
    # Get apply and update functions for actor and critic networks.
    actor_apply_fn, critic_apply_fn = apply_fns
    actor_update_fn, critic_update_fn = update_fns

    def _update_step(learner_state_with_learnable_instances: Tuple[LearnerState, EnvInstance], _: Any) -> Tuple[Tuple[LearnerState, EnvInstance], Tuple]:
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
            _ (Any): The current metrics info.

        """

        def _env_step(
            learner_state_with_start_state: LearnerStateWithStartState, _: Any
        ) -> Tuple[LearnerStateWithStartState, Tuple[PPOTransition, Metrics]]:
            """Step the environment."""
            learner_state, start_state = learner_state_with_start_state
            params, opt_states, key, env_state, last_timestep, last_done = learner_state

            # Select action
            key, policy_key = jax.random.split(key)
            actor_policy = actor_apply_fn(params.actor_params, last_timestep.observation)
            value = critic_apply_fn(params.critic_params, last_timestep.observation)

            action = actor_policy.sample(seed=policy_key)
            log_prob = actor_policy.log_prob(action)

            # Step environment
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0, 0))(env_state, action, start_state)

            done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)
            transition = PPOTransition(
                last_done, action, value, timestep.reward, log_prob, last_timestep.observation
            )
            learner_state = LearnerState(params, opt_states, key, env_state, timestep, done)
            learner_state_with_start_state = LearnerStateWithStartState(learner_state, start_state)
            metrics = timestep.extras["episode_metrics"] | timestep.extras["env_metrics"]
            return learner_state_with_start_state, (transition, metrics)

        # Sample learnable states and random states
        learner_state, learnable_instances = learner_state_with_learnable_instances
        key, sampled_key, gen_key = jax.random.split(learner_state.key, 3)
        sampled_key_0, sampled_key_1 = jax.random.split(sampled_key, 2)
        sampled_idxs = jax.random.choice(sampled_key_0, jnp.arange(config.ued.batch_size * config.ued.num_batches), (config.ued.num_sampled,), replace=False)
        sampled_keys = jax.random.split(sampled_key_1, config.ued.num_sampled)
        env_instances_sampled = jax.tree_util.tree_map(lambda x: x[sampled_idxs], learnable_instances)
        env_state_sampled, timestep_sampled = jax.vmap(env.set_env_instance, in_axes=(0,0))(
            env_instances_sampled, sampled_keys
        )

        gen_keys = jax.random.split(gen_key, config.arch.num_envs - config.ued.num_sampled)
        env_state_gen, timestep_gen = jax.vmap(env.reset)(gen_keys)
        timestep = jax.tree_util.tree_map(lambda x,y: jnp.concatenate([x,y], axis=0), timestep_gen, timestep_sampled)
        env_state = jax.tree_util.tree_map(
            lambda x, y: jnp.concatenate([x, y], axis=0),
            env_state_gen,
            env_state_sampled,
        )
        start_state = env_state
        learner_state = LearnerState(learner_state.params, learner_state.opt_states, learner_state.key, env_state, timestep, learner_state.dones)
        learner_state_with_start_state = LearnerStateWithStartState(learner_state, start_state)

        # Step environment for rollout length
        learner_state_with_start_state, (traj_batch, episode_metrics) = jax.lax.scan(
            _env_step, learner_state_with_start_state, None, config.system.rollout_length
        )
        learner_state, _ = learner_state_with_start_state

        # Calculate advantage
        params, opt_states, key, env_state, last_timestep, last_done = learner_state
        last_val = critic_apply_fn(params.critic_params, last_timestep.observation)

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

                    total_actor_loss = actor_loss - config.system.ent_coef * entropy
                    return total_actor_loss, (actor_loss, entropy)

                def _critic_loss_fn(
                    critic_params: FrozenDict,
                    traj_batch: PPOTransition,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the critic loss."""
                    # Rerun network
                    value = critic_apply_fn(critic_params, traj_batch.obs)

                    # Clipped MSE loss
                    value_pred_clipped = traj_batch.value + (value - traj_batch.value).clip(
                        -config.system.clip_eps, config.system.clip_eps
                    )
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

                    total_value_loss = config.system.vf_coef * value_loss
                    return total_value_loss, value_loss

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

            # Shuffle data and create minibatches
            batch_size = config.system.rollout_length * config.arch.num_envs
            permutation = jax.random.permutation(shuffle_key, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = tree.map(lambda x: merge_leading_dims(x, 2), batch)
            shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
            minibatches = tree.map(
                lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
                shuffled_batch,
            )

            # Update minibatches
            (params, opt_states, entropy_key), loss_info = jax.lax.scan(
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
        learner_state = LearnerState(params, opt_states, key, env_state, last_timestep, last_done)
        learner_state_with_learnable_instances = (learner_state, learnable_instances)
        return learner_state_with_learnable_instances, (episode_metrics, loss_info)

    def learner_fn(learner_state_with_learnable_instances: Tuple[LearnerState, EnvInstance]) -> ExperimentOutput[LearnerState]:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
        ----
            learner_state (NamedTuple):
                - params (Params): The initial model parameters.
                - opt_states (OptStates): The initial optimizer state.
                - key (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.

        """
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state_with_learnable_instances, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state_with_learnable_instances, None, config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state_with_learnable_instances[0],
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


@partial(jax.jit, static_argnums=(2, 3, 4))
def get_learnability_set(rng, actor_params, actor_apply_fn, config, env: JaxMarlWrapper):
    def _batch_step(_, rng):
        def _env_step(runner_state, _: Any) -> Tuple[LearnerState, SFLTransition]:
            """Step the environment."""
            rng, env_state, obs, start_state = runner_state

            # SELECT ACTION
            rng, policy_rng = jax.random.split(rng)
            actor_policy = actor_apply_fn(actor_params, obs)
            action = actor_policy.sample(seed=policy_rng)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step)(env_state, action, start_state)

            # LOG EPISODE METRICS
            done = jnp.repeat(timestep.last(), config["system"]["num_agents"])
            goal_reached = jnp.ravel(timestep.extras["env_metrics"]["GoalR"])

            transition = SFLTransition(
                done, goal_reached
            )
            runner_state = (rng, env_state, timestep.observation, start_state)
            return runner_state, transition

        @partial(jax.vmap, in_axes=(None, 1, 1))
        @partial(jax.jit, static_argnums=(0,))
        def _calc_outcomes_by_agent(max_steps: int, dones, goal_reached):
            idxs = jnp.arange(max_steps)

            @partial(jax.vmap, in_axes=(0, 0))
            def __ep_outcomes(start_idx, end_idx):
                mask = (
                    (idxs > start_idx) & (idxs <= end_idx) & (end_idx != max_steps)
                )
                success = jnp.sum(goal_reached * mask)
                return success

            done_idxs = jnp.argwhere(dones, size=10, fill_value=max_steps).squeeze()
            mask_done = jnp.where(done_idxs == max_steps, 0, 1)
            success = __ep_outcomes(
                jnp.concatenate([jnp.array([-1]), done_idxs[:-1]]), done_idxs
            )

            return {
                "success_rate": success.mean(where=mask_done),
            }

        # sample envs
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config.ued.batch_size)
        env_state, timestep= jax.vmap(env.reset)(reset_rng)
        env_instances = EnvInstance(
            agent_pos=env_state.env_state.state.pos,
            agent_theta=env_state.env_state.state.theta,
            goal_pos=env_state.env_state.state.goal,
            map_data=env_state.env_state.state.map_data,
            rew_lambda=env_state.env_state.state.rew_lambda,
        )
        runner_state = (rng, env_state, timestep.observation, env_state)
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config.ued.rollout_steps
        )
        print("traj batch done", traj_batch.done.shape)
        print("traj batch gr", traj_batch.goal_reached.shape)
        o = _calc_outcomes_by_agent(
            config.ued.rollout_steps,
            traj_batch.done,
            traj_batch.goal_reached,
        )
        print("o", o)
        success_by_env = o["success_rate"].reshape(
            (env.num_agents, config.ued.batch_size)
        )
        learnability_by_env = (success_by_env * (1 - success_by_env)).sum(axis=0)
        print("learnability_by_env", learnability_by_env)
        return None, (learnability_by_env, env_instances)
    
    rngs = jax.random.split(rng, config.ued.num_batches)
    _, (learnability, env_instances) = jax.lax.scan(
        _batch_step, None, rngs, config.ued.num_batches
    )

    flat_env_instances = jax.tree_map(
        lambda x: x.reshape((-1,) + x.shape[2:]), env_instances
    )
    learnability = learnability.flatten()
    top_1000 = jnp.argsort(learnability)[-config.ued.num_to_save :]
    print("top 1000", top_1000)

    top_1000_instances = jax.tree_map(
        lambda x: x.at[top_1000].get(), flat_env_instances
    )
    print("top 1000 instances", top_1000_instances)
    return learnability.at[top_1000].get(), top_1000_instances


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[LearnerState], Actor, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())

    # Get number of agents.
    config.system.num_agents = env.num_agents

    # PRNG keys.
    key, actor_net_key, critic_net_key = keys

    # Define network and optimiser.
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
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

    # Initialise observation with obs of all agents.
    obs = env.observation_spec.generate_value()
    init_x = tree.map(lambda x: x[jnp.newaxis, ...], obs)

    # Initialise actor params and optimiser state.
    actor_params = actor_network.init(actor_net_key, init_x)
    actor_opt_state = actor_optim.init(actor_params)

    # Initialise critic params and optimiser state.
    critic_params = critic_network.init(critic_net_key, init_x)
    critic_opt_state = critic_optim.init(critic_params)

    # Pack params.
    params = Params(actor_params, critic_params)

    # Pack apply and update functions.
    apply_fns = (actor_network.apply, critic_network.apply)
    update_fns = (actor_optim.update, critic_optim.update)

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, apply_fns, update_fns, config)
    learn = jax.pmap(learn, axis_name="device")

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
    dones = jnp.zeros(
        (config.arch.num_envs, config.system.num_agents),
        dtype=bool,
    )
    key, step_keys = jax.random.split(key)
    opt_states = OptStates(actor_opt_state, critic_opt_state)
    replicate_learner = (params, opt_states, step_keys, dones)

    # Duplicate learner for update_batch_size.
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)

    # Duplicate learner across devices.
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialise learner state.
    params, opt_states, step_keys, dones = replicate_learner
    init_learner_state = LearnerState(params, opt_states, step_keys, env_states, timesteps, dones)

    return learn, actor_network, init_learner_state


def run_experiment(_config: DictConfig) -> float:
    """Runs experiment."""
    _config.logger.system_name = "ff_ippo"
    config = copy.deepcopy(_config)

    n_devices = len(jax.devices())

    # Create the enviroments for train and eval.
    env, eval_env = make_env(config)

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
    eval_act_fn = make_ff_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    # Calculate total timesteps.
    config = check_total_timesteps(config)
    assert (
        config.system.num_updates > config.arch.num_evaluation
    ), "Number of updates per evaluation must be less than total number of updates."

    assert (
        config.arch.num_envs % config.system.num_minibatches == 0
    ), "Number of envs must be divisibile by number of minibatches."

    # Calculate number of updates per evaluation.
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
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

    # Run experiment for a total number of evaluations.
    max_episode_return = -jnp.inf
    best_params = None
    for eval_step in range(config.arch.num_evaluation):
        # Train.
        start_time = time.time()

        key, learnable_key = jax.random.split(key)
        learnabilities, learnable_instances = get_learnability_set(learnable_key, unreplicate_n_dims(learner_state.params.actor_params), actor_network.apply, config, env)
        broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
        replicate_learnable_instances = tree.map(broadcast, learnable_instances)

        # Duplicate learnable states across devices.
        replicate_learnable_instances = flax.jax_utils.replicate(replicate_learnable_instances, devices=jax.devices())

        learner_state_with_learnable_instances = (learner_state, replicate_learnable_instances)
        learner_output = learn(learner_state_with_learnable_instances)
        jax.block_until_ready(learner_output)

        # Log the results of the training.
        elapsed_time = time.time() - start_time
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / elapsed_time

        # Separately log timesteps, actoring metrics and training metrics.
        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:  # only log episode metrics if an episode was completed in the rollout.
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Prepare for evaluation.
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        key_e, *eval_keys = jax.random.split(key_e, n_devices + 1)
        eval_keys = jnp.stack(eval_keys)
        eval_keys = eval_keys.reshape(n_devices, -1)
        # Evaluate.
        eval_metrics = evaluator(trained_params, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
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
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)

        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})

        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger.
    logger.stop()

    return eval_performance


def test_get_learnability_set(_config: DictConfig) -> None:
    """Tests get_learnability_set."""
    _config.logger.system_name = "ff_ippo"
    config = copy.deepcopy(_config)

    # Create the enviroments for train and eval.
    env, eval_env = make_env(config)

    # PRNG keys.
    key, key_e, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner.
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )

    single_actor_params = unreplicate_n_dims(learner_state.params.actor_params)
    learnability, top_1000_instances = get_learnability_set(key, single_actor_params, actor_network.apply, config, env)


@hydra.main(
    config_path="../../../configs/default",
    config_name="ff_ippo.yaml",
    version_base="1.2",
)
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}IPPO SFL experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
