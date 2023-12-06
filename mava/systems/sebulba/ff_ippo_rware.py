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

from mava.utils.sebulba_tools import configure_computation_environment

configure_computation_environment()

import queue
import threading
import time
from collections import deque
from functools import partial
from typing import Any, Callable, Dict, List, Sequence, Tuple

import distrax
import flax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from chex import Array, PRNGKey
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import constant, orthogonal
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.evaluator import get_evaluator_setup
from mava.logger import SebulbaLogFn, logger_setup
from mava.types import OptStates, Params
from mava.types import SebulbaLearnerState as LearnerState
from mava.types import SebulbaTransition as Transition
from mava.utils.jax import convert_data, merge_leading_dims
from mava.wrappers.robot_warehouse import make_env


class Actor(nn.Module):
    """Actor Network."""

    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, observation: Array, action_mask: Array) -> distrax.Categorical:
        """Forward pass."""
        x = observation

        actor_output = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        actor_output = nn.relu(actor_output)
        actor_output = nn.Dense(256, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            actor_output
        )
        actor_output = nn.relu(actor_output)
        actor_output = nn.LayerNorm(use_scale=False)(actor_output)
        actor_output = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_output)
        masked_logits = jnp.where(
            action_mask,
            actor_output,
            jnp.finfo(jnp.float32).min,
        )
        actor_policy = distrax.Categorical(logits=masked_logits)

        return actor_policy


class Critic(nn.Module):
    """Critic Network."""

    @nn.compact
    def __call__(self, observation: Array) -> Array:
        """Forward pass."""

        critic_output = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            observation
        )
        critic_output = nn.relu(critic_output)
        critic_output = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            critic_output
        )
        critic_output = nn.relu(critic_output)
        critic_output = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic_output
        )

        return jnp.squeeze(critic_output, axis=-1)


def rollout(  # noqa: CCR001
    key: PRNGKey,
    cfg: DictConfig,
    rollout_queue: queue.Queue,
    params_queue: queue.Queue,
    learner_devices: List,
    device_thread_id: int,
    apply_fns: Tuple,
    log_fn: Callable,
    log_frequency: int,
) -> None:
    """Executor rollout loop."""
    # Create envs
    envs = make_env(
        num_envs=cfg.arch.num_envs,
        config=cfg,
    )()

    # Setup
    len_executor_device_ids = len(cfg.arch.executor_device_ids)
    t_env = 0
    start_time = time.time()

    # Get the apply functions for the actor and critic networks.
    vmap_actor_apply, vmap_critic_apply = apply_fns

    # Define the util functions: select action function and prepare data to share it with learner.
    @jax.jit
    def get_action_and_value(
        params: FrozenDict,
        next_obs: Array,
        actions_mask: Array,
        key: PRNGKey,
    ) -> Tuple:
        """Get action and value."""
        key, subkey = jax.random.split(key)

        policy = vmap_actor_apply(params.actor_params, next_obs, actions_mask)
        action, logprob = policy.sample_and_log_prob(seed=subkey)

        value = vmap_critic_apply(params.critic_params, next_obs).squeeze()
        return action, logprob, value, key

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        """Prepare data to share with learner."""
        return jax.tree_map(  # type: ignore
            lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage
        )

    # Define the episode info
    env_id = np.arange(cfg.arch.num_envs)
    episode_returns = np.zeros(
        (cfg.arch.num_envs,), dtype=np.float32
    )  # Accumulated episode returns
    returned_episode_returns = np.zeros(
        (cfg.arch.num_envs,), dtype=np.float32
    )  # Final episode returns
    episode_lengths = np.zeros(
        (cfg.arch.num_envs,), dtype=np.float32
    )  # Accumulated episode lengths
    returned_episode_lengths = np.zeros(
        (cfg.arch.num_envs,), dtype=np.float32
    )  # Final episode lengths

    # Define the data structure
    params_queue_get_time: deque = deque(maxlen=10)
    rollout_time: deque = deque(maxlen=10)
    rollout_queue_put_time: deque = deque(maxlen=10)

    # Reset envs
    next_obs, infos = envs.reset()
    next_dones = jnp.zeros((cfg.arch.num_envs, cfg.system.num_agents), dtype=jax.numpy.bool_)

    # Loop till the learner has finished training
    for update in range(1, cfg.system.num_updates + 2):
        # Setup
        env_recv_time: float = 0
        inference_time: float = 0
        storage_time: float = 0
        env_send_time: float = 0

        # Get the latest parameters from the learner
        params_queue_get_time_start = time.time()
        if cfg.arch.concurrency:
            if update != 2:
                params = params_queue.get()
                params.network_params["params"]["Dense_0"]["kernel"].block_until_ready()
        else:
            params = params_queue.get()
        params_queue_get_time.append(time.time() - params_queue_get_time_start)

        # Rollout
        rollout_time_start = time.time()
        storage: List = []
        # Loop over the rollout length
        for _ in range(0, cfg.system.rollout_length):
            # Get previous step info
            cached_next_obs = next_obs
            cached_next_dones = next_dones
            cached_infos = infos
            actions_mask = np.stack(cached_infos["actions_mask"])

            # Increment current timestep
            t_env += cfg.arch.n_threads_per_executor * len_executor_device_ids

            # Get action and value
            inference_time_start = time.time()
            (
                action,
                logprob,
                value,
                key,
            ) = get_action_and_value(params, cached_next_obs, actions_mask, key)
            inference_time += time.time() - inference_time_start

            # Step the environment
            env_send_time_start = time.time()
            cpu_action = np.array(action)
            next_obs, next_reward, terminated, truncated, infos = envs.step(cpu_action)
            next_done = terminated + truncated
            terminateds, truncateds, next_dones = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, cfg.system.num_agents).reshape(cfg.arch.num_envs, -1),
                (terminated, truncated, next_done),
            )

            # Append data to storage
            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()
            storage.append(
                Transition(
                    obs=cached_next_obs,
                    dones=cached_next_dones,
                    actions=action,
                    logprobs=logprob,
                    values=value,
                    rewards=next_reward,
                    truncations=truncateds,
                    terminations=terminateds,
                    actions_mask=actions_mask,
                )
            )
            storage_time += time.time() - storage_time_start

            # Update episode info
            episode_returns[env_id] += np.mean(next_reward)
            returned_episode_returns[env_id] = np.where(
                next_done,
                episode_returns[env_id],
                returned_episode_returns[env_id],
            )
            episode_returns[env_id] *= (1 - next_done) * (1 - truncated)
            episode_lengths[env_id] += 1
            returned_episode_lengths[env_id] = np.where(
                next_done,
                episode_lengths[env_id],
                returned_episode_lengths[env_id],
            )
            episode_lengths[env_id] *= (1 - next_done) * (1 - truncated)
        rollout_time.append(time.time() - rollout_time_start)

        # Prepare data to share with learner
        partitioned_storage = prepare_data(storage)
        sharded_storage = Transition(
            *list(  # noqa: C417
                map(
                    lambda x: jax.device_put_sharded(x, devices=learner_devices),  # type: ignore
                    partitioned_storage,
                )
            )
        )
        sharded_next_obs = jax.device_put_sharded(
            np.split(next_obs, len(learner_devices)), devices=learner_devices
        )
        sharded_next_done = jax.device_put_sharded(
            np.split(next_dones, len(learner_devices)), devices=learner_devices
        )
        payload = (
            t_env,
            sharded_storage,
            sharded_next_obs,
            sharded_next_done,
            np.mean(params_queue_get_time),
        )

        # Put data in the rollout queue to share it with the learner
        rollout_queue_put_time_start = time.time()
        rollout_queue.put(payload)
        rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)

        if (update % log_frequency == 0) or (cfg.system.num_updates + 1 == update):
            # Log info
            log_fn(
                log_type={"Executor": {"device_thread_id": device_thread_id}},
                t_env=t_env,
                metrics_to_log={
                    "episode_info": {
                        "episode_return": returned_episode_returns,
                        "episode_length": returned_episode_lengths,
                    },
                    "speed_info": {
                        "sps": int(t_env / (time.time() - start_time)),
                        "rollout_time": np.mean(rollout_time),
                    },
                    "queue_info": {
                        "params_queue_get_time": np.mean(params_queue_get_time),
                        "env_recv_time": env_recv_time,
                        "inference_time": inference_time,
                        "storage_time": storage_time,
                        "env_send_time": env_send_time,
                        "rollout_queue_put_time": np.mean(rollout_queue_put_time),
                    },
                },
            )


def get_trainer(cfg: DictConfig, apply_fns: Tuple, optim_updates_fn: Tuple) -> Callable:
    """Get the trainer function."""

    # Get the apply functions for the actor and critic networks.
    vmap_actor_apply, vmap_critic_apply = apply_fns
    update_actor_optim, update_critic_optim = optim_updates_fn

    # Define the function to compute GAE.
    def compute_gae_once(carry: Array, inp: Tuple, gamma: float, gae_lambda: float) -> Tuple:
        """Compute GAE once."""
        advantages = carry
        nextdone, nextvalues, curvalues, reward = inp
        nextnonterminal = 1.0 - nextdone

        delta = reward + gamma * nextvalues * nextnonterminal - curvalues
        advantages = delta + gamma * gae_lambda * nextnonterminal * advantages
        return advantages, advantages

    compute_gae_once = partial(
        compute_gae_once, gamma=cfg.system.gamma, gae_lambda=cfg.system.gae_lambda
    )
    vmap_compute_gae = jax.vmap(compute_gae_once, in_axes=(1, (1, 1, 1, 1)), out_axes=(1, 1))

    def compute_gae(
        agents_state: LearnerState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Transition,
    ) -> Tuple:
        """Compute GAE."""
        next_value = vmap_critic_apply(
            agents_state.params.critic_params,
            next_obs,
        )

        advantages = jnp.zeros(
            (
                cfg.arch.num_envs
                * cfg.arch.n_threads_per_executor
                * len(cfg.arch.executor_device_ids)
                // len(cfg.arch.learner_device_ids),
                cfg.system.num_agents,
            )
        )
        dones = jnp.concatenate([storage.dones, next_done[None, :]], axis=0)
        values = jnp.concatenate([storage.values, next_value[None, :]], axis=0)
        _, advantages = jax.lax.scan(
            vmap_compute_gae,
            advantages,
            (dones[1:], values[1:], values[:-1], storage.rewards),
            reverse=True,
        )
        return advantages, advantages + storage.values

    def actor_loss(
        actor_params: FrozenDict,
        obs: Array,
        actions_mask: Array,
        actions: Array,
        behavior_logprobs: Array,
        advantages: Array,
    ) -> Tuple:
        """Actor loss."""
        # Compute logprobs and entropy
        policy = vmap_actor_apply(actor_params, obs, actions_mask)
        newlogprob = policy.log_prob(actions)
        entropy = policy.entropy()

        logratio = newlogprob - behavior_logprobs
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1 - cfg.system.clip_eps, 1 + cfg.system.clip_eps)
        pg_loss = jnp.maximum(pg_loss1, pg_loss2).mean()

        entropy_loss = entropy.mean()
        loss = pg_loss - cfg.system.ent_coef * entropy_loss
        return loss, (pg_loss, entropy_loss, jax.lax.stop_gradient(approx_kl))

    def critic_loss(critic_params: FrozenDict, obs: Array, target_values: Array) -> Tuple:
        """Critic loss."""
        newvalue = vmap_critic_apply(critic_params, obs)

        # Value loss
        v_loss = 0.5 * ((newvalue - target_values) ** 2).mean()
        loss = v_loss * cfg.system.vf_coef
        return loss, (v_loss)

    def single_device_update(
        agents_state: LearnerState,
        sharded_storages: List,
        sharded_next_obs: List,
        sharded_next_done: List,
        key: jax.random.PRNGKey,
    ) -> Tuple:
        """Single device update."""
        # Horizontal stack all the data from different devices
        storage = jax.tree_map(lambda *x: jnp.hstack(x), *sharded_storages)

        # Define loss functions
        actor_loss_grad_fn = jax.value_and_grad(actor_loss, has_aux=True)
        critic_loss_grad_fn = jax.value_and_grad(critic_loss, has_aux=True)

        # Unpack data and compute GAE
        next_obs = jnp.concatenate(sharded_next_obs)
        next_done = jnp.concatenate(sharded_next_done)
        local_advantages, target_values = compute_gae(agents_state, next_obs, next_done, storage)

        # Normalize advantages across environments and agents
        if cfg.system.norm_adv:
            # Gather advantages across devices
            all_advantages = jax.lax.all_gather(local_advantages, axis_name="local_devices")
            advantages = jnp.concatenate(all_advantages, axis=1)

            # Normalize advantages across environments and agents
            mean_advantages = advantages.mean((1, 2), keepdims=True)
            std_advantages = advantages.std((1, 2), keepdims=True)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)

            # Split advantages across devices
            split_advantages = jnp.split(advantages, all_advantages.shape[0], axis=1)
            local_advantages = split_advantages[jax.process_index()]

        def update_epoch(
            carry: Tuple[LearnerState, PRNGKey], _: Any
        ) -> Tuple[Tuple[LearnerState, PRNGKey], Tuple]:
            """Update epoch."""
            # Unpack data and generate keys
            agents_state, key = carry
            key, subkey = jax.random.split(key)

            # Shuffle data
            def flatten_and_shuffle(
                storage: Transition, local_advantages: Array, target_values: Array
            ) -> Tuple[Transition, Array, Array]:
                """Flatten and shuffle data."""
                flatten = lambda x: merge_leading_dims(x)
                convert = lambda x: convert_data(
                    x, subkey, cfg.system.num_minibatches, cfg.system.gradient_accumulation_steps
                )

                shuffled_storage = jax.tree_map(lambda x: convert(flatten(x)), storage)
                shuffled_advantages = convert(flatten(local_advantages))
                shuffled_target_values = convert(flatten(target_values))

                return shuffled_storage, shuffled_advantages, shuffled_target_values

            vmap_flatten_and_shuffle = jax.vmap(flatten_and_shuffle, in_axes=(2), out_axes=(2))
            (
                shuffled_storage,
                shuffled_advantages,
                shuffled_target_values,
            ) = vmap_flatten_and_shuffle(storage, local_advantages, target_values)

            def update_minibatch(
                agents_state: LearnerState, minibatch: Tuple
            ) -> Tuple[LearnerState, Tuple]:
                """Update minibatch."""
                # Unpack data
                (
                    mb_shuffled_storage,
                    mb_advantages,
                    mb_target_values,
                ) = minibatch
                mb_obs, mb_actions_mask, mb_actions, mb_behavior_logprobs = (
                    mb_shuffled_storage.obs,
                    mb_shuffled_storage.actions_mask,
                    mb_shuffled_storage.actions,
                    mb_shuffled_storage.logprobs,
                )

                # Compute losses and gradients
                (p_loss, (pg_loss, entropy_loss, approx_kl),), actor_grads = actor_loss_grad_fn(
                    agents_state.params.actor_params,
                    mb_obs,
                    mb_actions_mask,
                    mb_actions,
                    mb_behavior_logprobs,
                    mb_advantages,
                )
                (c_loss, (v_loss)), critic_grads = critic_loss_grad_fn(
                    agents_state.params.critic_params,
                    mb_obs,
                    mb_target_values,
                )
                loss = p_loss + c_loss

                # Compute mean loss across all devices
                actor_grads = jax.lax.pmean(actor_grads, axis_name="local_devices")
                critic_grads = jax.lax.pmean(critic_grads, axis_name="local_devices")

                # Update actor params and optimiser state
                actor_updates, actor_new_opt_state = update_actor_optim(
                    actor_grads, agents_state.opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(
                    agents_state.params.actor_params, actor_updates
                )

                # Update critic params and optimiser state
                critic_updates, critic_new_opt_state = update_critic_optim(
                    critic_grads, agents_state.opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(
                    agents_state.params.critic_params, critic_updates
                )

                # Pack new params and optimiser state
                new_params = Params(actor_new_params, critic_new_params)
                new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)
                agents_state = LearnerState(params=new_params, opt_states=new_opt_state)

                return agents_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl)

            # Loop through minibatches
            agents_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl,) = jax.lax.scan(
                update_minibatch,
                agents_state,
                (
                    shuffled_storage,
                    shuffled_advantages,
                    shuffled_target_values,
                ),
            )
            return (agents_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl)

        # Loop through epochs
        (agents_state, key), (
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
        ) = jax.lax.scan(update_epoch, (agents_state, key), (), length=cfg.system.ppo_epochs)

        # Compute mean loss across all devices
        loss_info = {
            "total_loss": loss,
            "loss_actor": pg_loss,
            "value_loss": v_loss,
            "entropy": entropy_loss,
            "approx_kl": approx_kl,
        }
        loss_info = jax.tree_map(lambda x: jax.lax.pmean(x, axis_name="local_devices"), loss_info)

        return agents_state, key, loss_info

    return single_device_update


def run_experiment(cfg: DictConfig) -> None:  # noqa: CCR001
    """Run experiment."""
    # Logger setup
    cfg_dict: Dict = OmegaConf.to_container(cfg, resolve=True)
    log: SebulbaLogFn = logger_setup(cfg_dict)  # type: ignore

    # Config setup
    cfg.system.batch_size = int(
        cfg.arch.num_envs
        * cfg.system.rollout_length
        * cfg.arch.n_threads_per_executor
        * len(cfg.arch.executor_device_ids)
    )
    cfg.system.minibatch_size = int(cfg.system.batch_size // cfg.system.num_minibatches)
    assert (
        cfg.arch.num_envs % len(cfg.arch.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(cfg.arch.num_envs / len(cfg.arch.learner_device_ids))
        * cfg.arch.n_threads_per_executor
        % cfg.system.num_minibatches
        == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
    cfg.system.total_timesteps = cfg.system.num_updates * (cfg.system.batch_size)
    local_devices = jax.local_devices()
    learner_devices = [local_devices[d_id] for d_id in cfg.arch.learner_device_ids]
    executor_devices = [local_devices[d_id] for d_id in cfg.arch.executor_device_ids]
    cfg.system.executor_devices = [str(item) for item in executor_devices]
    cfg.system.learner_devices = [str(item) for item in learner_devices]
    log_frequency = cfg.system.num_updates // cfg.arch.num_evaluation
    pprint(cfg_dict)

    # PRNG keys.
    key = jax.random.PRNGKey(cfg.system.seed)
    key, eval_key, actor_key, critic_key = jax.random.split(key, 4)
    learner_keys = jax.device_put_replicated(key, learner_devices)

    # Define network and optimiser
    def get_linear_schedule_fn(lr: float) -> Callable:
        def linear_schedule(count: int) -> float:
            """Linear learning rate schedule"""
            frac: float = (
                1.0
                - (count // (cfg.system.num_minibatches * cfg.system.ppo_epochs))
                / cfg.system.num_updates
            )
            return lr * frac

        return linear_schedule

    dummy_envs = make_env(  # Create dummy_envs to get observation and action spaces
        num_envs=cfg.arch.num_envs,
        config=cfg,
    )()
    cfg.system.num_agents = dummy_envs.single_observation_space.shape[0]
    cfg.system.num_actions = int(dummy_envs.single_action_space.nvec[0])
    cfg.system.single_obs_dim = dummy_envs.single_observation_space.shape[1]
    init_obs = np.array([dummy_envs.single_observation_space.sample()[0]])
    init_action_mask = np.ones((1, cfg.system.num_actions))
    dummy_envs.close()  # close dummy envs

    actor = Actor(action_dim=cfg.system.num_actions)
    actor_params = actor.init(actor_key, init_obs, init_action_mask)
    actor_optim = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(cfg.system.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=get_linear_schedule_fn(cfg.system.actor_lr)
                if cfg.system.anneal_lr
                else cfg.system.actor_lr,
                eps=1e-5,
            ),
        ),
        every_k_schedule=cfg.system.gradient_accumulation_steps,
    )
    actor_opt_state = actor_optim.init(actor_params)
    vmap_actor_apply = jax.vmap(actor.apply, in_axes=(None, 1, 1), out_axes=(1))

    critic = Critic()
    critic_params = critic.init(critic_key, init_obs)
    critic_optim = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(cfg.system.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=get_linear_schedule_fn(cfg.system.critic_lr)
                if cfg.system.anneal_lr
                else cfg.system.critic_lr,
                eps=1e-5,
            ),
        ),
        every_k_schedule=cfg.system.gradient_accumulation_steps,
    )
    critic_opt_state = critic_optim.init(critic_params)
    vmap_critic_apply = jax.vmap(critic.apply, in_axes=(None, 1), out_axes=(1))

    # Define agents state
    agents_state = LearnerState(
        params=Params(
            actor_params=actor_params,
            critic_params=critic_params,
        ),
        opt_states=OptStates(
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
        ),
    )
    # Replicate agents state per learner device
    agents_state = flax.jax_utils.replicate(agents_state, devices=learner_devices)
    unreplicated_params = flax.jax_utils.unreplicate(agents_state.params)

    # Learner Setup
    single_device_update = get_trainer(
        cfg, (vmap_actor_apply, vmap_critic_apply), (actor_optim.update, critic_optim.update)
    )
    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=learner_devices,
    )

    # Evaluation setup
    evaluator_setup = get_evaluator_setup(cfg.arch.arch_name)
    eval_fn = evaluator_setup(cfg, vmap_actor_apply, log)

    # Executor Setup
    params_queues: List = []
    rollout_queues: List = []
    # Loop through each executor device
    for d_idx, d_id in enumerate(cfg.arch.executor_device_ids):
        # Replicate params per executor device
        device_params = jax.device_put(unreplicated_params, local_devices[d_id])
        # Loop through each executor thread
        for thread_id in range(cfg.arch.n_threads_per_executor):
            params_queues.append(queue.Queue(maxsize=1))
            rollout_queues.append(queue.Queue(maxsize=1))
            params_queues[-1].put(device_params)
            threading.Thread(
                target=rollout,
                args=(
                    jax.device_put(key, local_devices[d_id]),
                    cfg,
                    rollout_queues[-1],
                    params_queues[-1],
                    learner_devices,
                    d_idx * cfg.arch.n_threads_per_executor + thread_id,
                    (vmap_actor_apply, vmap_critic_apply),
                    log,
                    log_frequency,
                ),
            ).start()

    rollout_queue_get_time: deque = deque(maxlen=10)
    data_transfer_time: deque = deque(maxlen=10)
    trainer_update_number = 0
    while True:
        trainer_update_number += 1
        rollout_queue_get_time_start = time.time()
        sharded_storages = []
        sharded_next_obss = []
        sharded_next_dones = []

        # Loop through each executor device
        for d_idx, _ in enumerate(cfg.arch.executor_device_ids):
            # Loop through each executor thread
            for thread_id in range(cfg.arch.n_threads_per_executor):
                # Get data from rollout queue
                (
                    t_env,
                    sharded_storage,
                    sharded_next_obs,
                    sharded_next_done,
                    avg_params_queue_get_time,
                ) = rollout_queues[d_idx * cfg.arch.n_threads_per_executor + thread_id].get()
                sharded_storages.append(sharded_storage)
                sharded_next_obss.append(sharded_next_obs)
                sharded_next_dones.append(sharded_next_done)

        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()

        # Update learner
        (agents_state, learner_keys, loss_info) = multi_device_update(
            agents_state,
            sharded_storages,
            sharded_next_obss,
            sharded_next_dones,
            learner_keys,
        )

        # Send updated params to executors
        unreplicated_params = flax.jax_utils.unreplicate(agents_state.params)
        for d_idx, d_id in enumerate(cfg.arch.executor_device_ids):
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
            for thread_id in range(cfg.arch.n_threads_per_executor):
                params_queues[d_idx * cfg.arch.n_threads_per_executor + thread_id].put(
                    device_params
                )

        if trainer_update_number % log_frequency == 0:
            # Logging training info
            log(
                log_type={"Learner": {"trainer_update_number": trainer_update_number}},
                t_env=t_env,
                metrics_to_log={
                    "loss_info": loss_info,
                    "queue_info": {
                        "rollout_queue_get_time": np.mean(rollout_queue_get_time),
                        "data_transfer_time": np.mean(data_transfer_time),
                        "rollout_params_queue_get_time_diff": np.mean(rollout_queue_get_time)
                        - avg_params_queue_get_time,
                        "rollout_queue_size": rollout_queues[0].qsize(),
                        "params_queue_size": params_queues[0].qsize(),
                    },
                    "speed_info": {
                        "training_time": time.time() - training_time_start,
                    },
                },
            )

            # Evaluation
            eval_key, _ = jax.random.split(eval_key)
            eval_fn(
                params=unreplicated_params,
                key=eval_key,
                t_env=t_env,
            )

        # Check if training is finished
        if trainer_update_number >= cfg.system.num_updates:
            eval_key, _ = jax.random.split(eval_key)
            eval_fn(
                params=unreplicated_params,
                key=eval_key,
                t_env=t_env,
            )
            break


@hydra.main(config_path="../../configs", config_name="default_ff_ippo.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> None:
    """Experiment entry point."""

    # Run experiment.
    run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}IPPO experiment completed{Style.RESET_ALL}")


if __name__ == "__main__":
    hydra_entry_point()
