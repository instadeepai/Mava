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
from typing import Any, Dict, Tuple

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
from rich.pretty import pprint

from mava.evaluator import get_eval_fn, make_ff_eval_act_fn
from mava.networks import FeedForwardActor as Actor
from mava.networks import FeedForwardValueNet as Critic
from mava.systems.ppo.types import LearnerState, OptStates, Params, PPOTransition
from mava.types import ActorApply, CriticApply, ExperimentOutput, LearnerFn, MarlEnv, Metrics
from mava.utils import make_env as environments
from mava.utils.checkpointing import Checkpointer
from mava.utils.config import check_anakin_ppo_config, check_total_timesteps
from mava.utils.jax_utils import merge_leading_dims, unreplicate_batch_dim, unreplicate_n_dims
from mava.utils.logger import LogEvent, MavaLogger
from mava.utils.multistep import calculate_gae
from mava.utils.network_utils import get_action_head
from mava.utils.sebulba import RecordTimeTo
from mava.utils.training import make_learning_rate
from mava.wrappers.episode_metrics import get_final_step_metrics

import flax.linen as nn


def env_step_fn(
    learner_state: LearnerState,
    _: Any,
    actor: nn.Module,
    critic: nn.Module,
    env: MarlEnv,
    config: DictConfig,
) -> Tuple[LearnerState, Tuple[PPOTransition, Metrics]]:
    """Step the environment and collect experience."""
    params, opt_states, key, env_state, last_timestep, last_done = learner_state

    # Select action
    key, policy_key = jax.random.split(key)
    actor_policy = actor.apply(params.actor_params, last_timestep.observation)
    value = critic.apply(params.critic_params, last_timestep.observation)
    action = actor_policy.sample(seed=policy_key)
    log_prob = actor_policy.log_prob(action)

    # Step environment
    env_state, timestep = jax.vmap(env.step)(env_state, action)

    done = timestep.last().repeat(env.num_agents).reshape(config.arch.num_envs, -1)
    transition = PPOTransition(
        last_done, action, value, timestep.reward, log_prob, last_timestep.observation
    )
    learner_state = LearnerState(params, opt_states, key, env_state, timestep, done)
    return learner_state, (transition, timestep.extras["episode_metrics"])


def actor_loss_fn(
    actor_params: FrozenDict,
    trajectory: PPOTransition,
    advantage: chex.Array,
    key: chex.PRNGKey,
    actor: nn.Module,
    clip_eps: float,
    ent_coef: float,
) -> Tuple:
    """Calculate the actor loss."""
    actor_policy = actor.apply(actor_params, trajectory.obs)
    log_prob = actor_policy.log_prob(trajectory.action)

    # Calculate actor loss
    ratio = jnp.exp(log_prob - trajectory.log_prob)
    # Normalize advantage at minibatch level
    advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    actor_loss1 = ratio * advantage
    actor_loss2 = ratio.clip(1 - clip_eps, 1 + clip_eps) * advantage
    actor_loss = -jnp.minimum(actor_loss1, actor_loss2).mean()

    entropy = actor_policy.entropy(seed=key).mean()
    total_actor_loss = actor_loss - ent_coef * entropy
    return total_actor_loss, (actor_loss, entropy)


def critic_loss_fn(
    critic_params: FrozenDict,
    trajectory: PPOTransition,
    targets: chex.Array,
    critic: nn.Module,
    clip_eps: float,
    vf_coef: float,
) -> Tuple:
    """Calculate the critic loss."""
    value = critic.apply(critic_params, trajectory.obs)

    # Clipped MSE loss
    value_pred_clipped = trajectory.value + (value - trajectory.value).clip(-clip_eps, clip_eps)
    value_losses = jnp.square(value - targets)
    value_losses_clipped = jnp.square(value_pred_clipped - targets)
    value_loss = 0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()

    total_value_loss = vf_coef * value_loss
    return total_value_loss, value_loss


def update_minibatch(
    train_state: Tuple,  # TODO: type
    batch: Tuple,  # TODO: type
    networks: Tuple[nn.Module, nn.Module],
    optims: Tuple[optax.GradientTransformationExtraArgs, optax.GradientTransformationExtraArgs],
    config: DictConfig,
) -> Tuple:
    """Update the network for a single minibatch."""
    actor, critic = networks
    actor_opt, critic_opt = optims
    params, opt_states, key = train_state
    trajectory, advantages, targets = batch

    # Calculate actor loss
    key, entropy_key = jax.random.split(key)
    actor_grad_fn = jax.value_and_grad(actor_loss_fn, has_aux=True)
    actor_loss_info, actor_grads = actor_grad_fn(
        params.actor_params,
        trajectory,
        advantages,
        entropy_key,
        actor,
        config.system.clip_eps,
        config.system.ent_coef,
    )

    # Calculate critic loss
    critic_grad_fn = jax.value_and_grad(critic_loss_fn, has_aux=True)
    value_loss_info, critic_grads = critic_grad_fn(
        params.critic_params,
        trajectory,
        targets,
        critic,
        config.system.clip_eps,
        config.system.vf_coef,
    )

    # Compute pmean over batch and devices
    mean_batch_and_device = lambda x: jax.lax.pmean(jax.lax.pmean(x, "batch"), "device")
    actor_grads, actor_loss_info = mean_batch_and_device((actor_grads, actor_loss_info))
    critic_grads, value_loss_info = mean_batch_and_device((critic_grads, value_loss_info))

    # Update params and optimizer state
    actor_updates, actor_new_opt_state = actor_opt.update(actor_grads, opt_states.actor_opt_state)
    actor_new_params = optax.apply_updates(params.actor_params, actor_updates)

    critic_updates, critic_new_opt_state = critic_opt.update(
        critic_grads, opt_states.critic_opt_state
    )
    critic_new_params = optax.apply_updates(params.critic_params, critic_updates)

    new_params = Params(actor_new_params, critic_new_params)
    new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)

    actor_loss, (_, entropy) = actor_loss_info
    value_loss, unscaled_value_loss = value_loss_info

    loss_info = {
        "total_loss": actor_loss + value_loss,
        "value_loss": unscaled_value_loss,
        "actor_loss": actor_loss,
        "entropy": entropy,
    }
    return (new_params, new_opt_state, entropy_key), loss_info


def update_epoch(
    update_state: Tuple,  # TODO: type
    _: Any,
    networks: Tuple[nn.Module, nn.Module],
    optims: Tuple[optax.GradientTransformationExtraArgs, optax.GradientTransformationExtraArgs],
    config: DictConfig,
) -> Tuple:
    """Update the network for a single epoch."""
    params, opt_states, trajectory, advantages, targets, key = update_state
    key, shuffle_key, entropy_key = jax.random.split(key, 3)

    # Shuffle data and create minibatches
    batch_size = config.system.rollout_length * config.arch.num_envs
    permutation = jax.random.permutation(shuffle_key, batch_size)
    batch = (trajectory, advantages, targets)
    batch = tree.map(lambda x: merge_leading_dims(x, 2), batch)
    shuffled_batch = tree.map(lambda x: jnp.take(x, permutation, axis=0), batch)
    minibatches = tree.map(
        lambda x: jnp.reshape(x, (config.system.num_minibatches, -1, *x.shape[1:])),
        shuffled_batch,
    )

    # Define partially applied update_minibatch function with fixed arguments
    update_fn = lambda state, batch: update_minibatch(state, batch, networks, optims, config)

    # Update minibatches
    (params, opt_states, entropy_key), loss_info = jax.lax.scan(
        update_fn, (params, opt_states, entropy_key), minibatches
    )

    update_state = (params, opt_states, trajectory, advantages, targets, key)
    return update_state, loss_info


def get_learner_fn(
    env: MarlEnv,
    networks: Tuple[nn.Module, nn.Module],
    optims: Tuple[optax.GradientTransformationExtraArgs, optax.GradientTransformationExtraArgs],
    config: DictConfig,
) -> LearnerFn[LearnerState]:
    """Get the learner function."""
    actor, critic = networks

    # Partially apply env_step_fn with fixed arguments
    env_step = lambda state, unused: env_step_fn(state, unused, actor, critic, env, config)

    # Define partially applied update_epoch function with fixed arguments
    epoch_update_fn = lambda state, unused: update_epoch(state, unused, networks, optims, config)

    def _update_step(learner_state: LearnerState, _: Any) -> Tuple[LearnerState, Tuple]:
        """A single update of the network."""
        # Step environment for rollout length
        learner_state, (trajectory, episode_metrics) = jax.lax.scan(
            env_step, learner_state, length=config.system.rollout_length
        )

        # Calculate advantage
        params, opt_states, key, env_state, last_timestep, last_done = learner_state
        last_val = critic.apply(params.critic_params, last_timestep.observation)
        advantages, targets = calculate_gae(
            trajectory, last_val, last_done, config.system.gamma, config.system.gae_lambda
        )

        # Update epochs
        update_state = (params, opt_states, trajectory, advantages, targets, key)
        update_state, loss_info = jax.lax.scan(
            epoch_update_fn, update_state, length=config.system.ppo_epochs
        )

        params, opt_states, _, _, _, key = update_state
        learner_state = LearnerState(params, opt_states, key, env_state, last_timestep, last_done)
        return learner_state, (episode_metrics, loss_info)

    def learner_fn(learner_state: LearnerState) -> ExperimentOutput[LearnerState]:
        """Learner function for updating network parameters."""
        batched_update_step = jax.vmap(_update_step, in_axes=(0, None), axis_name="batch")

        learner_state, (episode_info, loss_info) = jax.lax.scan(
            batched_update_step, learner_state, length=config.system.num_updates_per_eval
        )
        return ExperimentOutput(
            learner_state=learner_state,
            episode_metrics=episode_info,
            train_metrics=loss_info,
        )

    return learner_fn


def learner_setup(
    env: MarlEnv, keys: chex.Array, config: DictConfig
) -> Tuple[LearnerFn[LearnerState], Actor, LearnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    n_devices = len(jax.devices())
    config.system.num_agents = env.num_agents
    key, actor_net_key, critic_net_key = keys

    # Define network
    actor_torso = hydra.utils.instantiate(config.network.actor_network.pre_torso)
    action_head, _ = get_action_head(env.action_spec)
    actor_action_head = hydra.utils.instantiate(action_head, action_dim=env.action_dim)
    critic_torso = hydra.utils.instantiate(config.network.critic_network.pre_torso)

    actor_network = Actor(torso=actor_torso, action_head=actor_action_head)
    critic_network = Critic(torso=critic_torso)

    # Setup optimisers
    actor_lr = make_learning_rate(config.system.actor_lr, config)
    critic_lr = make_learning_rate(config.system.critic_lr, config)
    actor_opt = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(actor_lr, eps=1e-5),
    )
    critic_opt = optax.chain(
        optax.clip_by_global_norm(config.system.max_grad_norm),
        optax.adam(critic_lr, eps=1e-5),
    )

    # Initialize networks and optimisers
    obs = env.observation_spec.generate_value()
    obs = tree.map(lambda x: x[jnp.newaxis, ...], obs)
    actor_params = actor_network.init(actor_net_key, obs)
    actor_opt_state = actor_opt.init(actor_params)
    critic_params = critic_network.init(critic_net_key, obs)
    critic_opt_state = critic_opt.init(critic_params)

    params = Params(actor_params, critic_params)
    opt_states = OptStates(actor_opt_state, critic_opt_state)

    # Load model from checkpoint if specified
    if config.logger.checkpointing.load_model:
        loaded_checkpoint = Checkpointer(
            model_name=config.logger.system_name,
            **config.logger.checkpointing.load_args,
        )
        restored_params, _ = loaded_checkpoint.restore_params(input_params=params)
        params = restored_params

    # Setup learner function
    networks = (actor_network, critic_network)
    optimisers = (actor_opt, critic_opt)
    learn = get_learner_fn(env, networks, optimisers, config)
    learn = jax.pmap(learn, axis_name="device")

    # Initialize environment states
    key, *env_keys = jax.random.split(
        key, n_devices * config.system.update_batch_size * config.arch.num_envs + 1
    )
    env_states, timesteps = jax.vmap(env.reset)(jnp.stack(env_keys))

    # Reshape states to (devices, update batch size, num_envs, ...)
    reshape_states = lambda x: x.reshape(
        (n_devices, config.system.update_batch_size, config.arch.num_envs) + x.shape[1:]
    )
    env_states = tree.map(reshape_states, env_states)
    timesteps = tree.map(reshape_states, timesteps)

    # Replicate learner state across devices and batches
    dones = jnp.zeros((config.arch.num_envs, config.system.num_agents), dtype=bool)
    key, step_keys = jax.random.split(key)

    replicate_learner = (params, opt_states, step_keys, dones)
    # Duplicate for update_batch_size
    broadcast = lambda x: jnp.broadcast_to(x, (config.system.update_batch_size, *x.shape))
    replicate_learner = tree.map(broadcast, replicate_learner)
    # Duplicate across devices
    replicate_learner = flax.jax_utils.replicate(replicate_learner, devices=jax.devices())

    # Initialize learner state
    params, opt_states, step_keys, dones = replicate_learner
    init_learner_state = LearnerState(params, opt_states, step_keys, env_states, timesteps, dones)

    return learn, actor_network, init_learner_state


def run_experiment(config: DictConfig) -> float:
    """Runs experiment."""
    config.logger.system_name = "ff_ippo"
    config.arch.devices = str(jax.devices())
    config.arch.n_devices = n_devices = len(jax.devices())

    # Config checks and additions
    config = check_total_timesteps(config)
    check_anakin_ppo_config(config)
    config.system.num_updates_per_eval = config.system.num_updates // config.arch.num_evaluation
    steps_per_rollout = (
        n_devices
        * config.system.num_updates_per_eval
        * config.system.rollout_length
        * config.system.update_batch_size
        * config.arch.num_envs
    )
    pprint(OmegaConf.to_container(config, resolve=True))

    # Create environments and setup PRNG keys
    env, eval_env = environments.make(config)
    key, eval_key, actor_net_key, critic_net_key = jax.random.split(
        jax.random.PRNGKey(config.system.seed), num=4
    )

    # Setup learner and evaluator
    learn, actor_network, learner_state = learner_setup(
        env, (key, actor_net_key, critic_net_key), config
    )
    eval_keys = jax.random.split(eval_key, n_devices)
    eval_act_fn = make_ff_eval_act_fn(actor_network.apply, config)
    evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=False)

    logger = MavaLogger(config)

    # Setup checkpointer if needed
    if config.logger.checkpointing.save_model:
        checkpointer = Checkpointer(
            metadata=config,
            model_name=config.logger.system_name,
            **config.logger.checkpointing.save_args,
        )

    # Run experiment for the specified number of evaluations
    max_episode_return = -jnp.inf
    best_params = None
    learn_time = []

    for eval_step in range(config.arch.num_evaluation):
        # Train
        with RecordTimeTo(learn_time):
            learner_output = learn(learner_state)
            jax.block_until_ready(learner_output)

        # Log training results
        t = int(steps_per_rollout * (eval_step + 1))
        episode_metrics, ep_completed = get_final_step_metrics(learner_output.episode_metrics)
        episode_metrics["steps_per_second"] = steps_per_rollout / learn_time.pop()

        logger.log({"timestep": t}, t, eval_step, LogEvent.MISC)
        if ep_completed:
            logger.log(episode_metrics, t, eval_step, LogEvent.ACT)
        logger.log(learner_output.train_metrics, t, eval_step, LogEvent.TRAIN)

        # Evaluate
        trained_params = unreplicate_batch_dim(learner_state.params.actor_params)
        eval_key, *eval_keys = jax.random.split(eval_key, n_devices + 1)
        eval_keys = jnp.stack(eval_keys).reshape(n_devices, -1)
        eval_metrics = evaluator(trained_params, eval_keys, {})
        logger.log(eval_metrics, t, eval_step, LogEvent.EVAL)
        episode_return = jnp.mean(eval_metrics["episode_return"])

        # Save checkpoint if enabled
        if config.logger.checkpointing.save_model:
            checkpointer.save(
                timestep=steps_per_rollout * (eval_step + 1),
                unreplicated_learner_state=unreplicate_n_dims(learner_output.learner_state),
                episode_return=episode_return,
            )

        # Track best performance
        if config.arch.absolute_metric and max_episode_return <= episode_return:
            best_params = copy.deepcopy(trained_params)
            max_episode_return = episode_return

        # Update learner state
        learner_state = learner_output.learner_state

    # Record final evaluation performance
    eval_performance = float(jnp.mean(eval_metrics[config.env.eval_metric]))

    # Measure absolute metric if needed
    if config.arch.absolute_metric:
        abs_metric_evaluator = get_eval_fn(eval_env, eval_act_fn, config, absolute_metric=True)
        eval_keys = jax.random.split(key, n_devices)
        eval_metrics = abs_metric_evaluator(best_params, eval_keys, {})
        t = int(steps_per_rollout * (eval_step + 1))
        logger.log(eval_metrics, t, eval_step, LogEvent.ABSOLUTE)

    # Stop the logger
    logger.stop()
    return eval_performance


@hydra.main(config_path="../../../configs/default", config_name="ff_ippo.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    OmegaConf.set_struct(cfg, False)  # Allow dynamic attributes
    eval_performance = run_experiment(cfg)
    print(f"{Fore.CYAN}{Style.BRIGHT}IPPO experiment completed{Style.RESET_ALL}")
    return eval_performance


if __name__ == "__main__":
    hydra_entry_point()
