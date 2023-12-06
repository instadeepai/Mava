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

import queue
import random
import threading
import time
import uuid
from collections import deque
from functools import partial
from types import SimpleNamespace
from typing import Callable, Dict, List, Sequence

import distrax
import flax
import flax.linen as nn
import gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import rware
from chex import Array
from colorama import Fore, Style
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import constant, orthogonal
from omegaconf import DictConfig, OmegaConf
from rich.pretty import pprint

from mava.logger import logger_setup
from mava.types import LearnerState, OptStates, Params
from mava.types import SebulbaTransition as Transition
from mava.utils.sebulba_tools import configure_computation_environment
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
        # TODO: make it a feature
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


# Evaluator


def evaluation(
    envs: gym.vector.AsyncVectorEnv,
    params: FrozenDict,
    vmap_actor_apply,
    key,
    cfg,
    log_fn,
    t_env,
) -> Dict[str, float]:

    # During evaluation, we want deterministic ordering of seeds
    @jax.jit
    def get_action_and_value(  # TODO: Use the power_tools action methods?
        params: flax.core.FrozenDict,
        next_obs: np.ndarray,
        actions_mask: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        policy = vmap_actor_apply(params.actor_params, next_obs, actions_mask)
        key, subkey = jax.random.split(key)
        raw_action, _ = policy.sample_and_log_prob(seed=subkey)

        return next_obs, raw_action, key

    # put data in the last index
    episode_returns = np.zeros((cfg.arch.num_eval_episodes,), dtype=np.float32)
    episode_lengths = np.zeros((cfg.arch.num_eval_episodes,), dtype=np.float32)
    next_obs, infos = envs.reset()
    next_done = jnp.zeros(cfg.arch.num_envs, dtype=jax.numpy.bool_)
    # TODO: make it flexible to different epiosdes lengths
    for _ in range(0, 500):
        cached_next_obs = next_obs
        cached_next_done = next_done
        cached_infos = infos
        actions_mask = np.stack(cached_infos["actions_mask"])
        (
            cached_next_obs,
            action,
            key,
        ) = get_action_and_value(params, cached_next_obs, actions_mask, key)
        cpu_action = np.array(action)
        next_obs, next_reward, next_done, _, infos = envs.step(cpu_action)
        env_id = np.arange(cfg.arch.num_eval_episodes)
        episode_returns[env_id] += np.mean(next_reward)
        episode_lengths[env_id] += 1
    # TODO: add sps
    log_fn(
        log_type={"Evaluator": {}},
        t_env=t_env,
        metrics_to_log={
            "episode_info": {
                "episode_return": episode_returns,
                "episode_length": np.zeros_like(episode_returns),
            },
        },
    )


def rollout(
    key: jax.random.PRNGKey,
    cfg,
    rollout_queue,
    params_queue: queue.Queue,
    learner_devices,
    device_thread_id,
    actor_device_id,
    log_fn,
):
    envs = make_env(
        num_envs=cfg.arch.num_envs,
        config=cfg,
    )()
    len_executor_device_ids = len(cfg.arch.executor_device_ids)
    global_step = 0
    start_time = time.time()

    # TODO: get this from main function
    actor_network = Actor(action_dim=cfg.system.num_actions)
    critic_network = Critic()
    vmap_actor_apply = jax.vmap(actor_network.apply, in_axes=(None, 1, 1), out_axes=(1))
    vmap_critic_apply = jax.vmap(critic_network.apply, in_axes=(None, 1), out_axes=(1))

    @jax.jit
    def get_action_and_value(  # TODO: Use the power_tools action methods?
        params: flax.core.FrozenDict,
        next_obs: np.ndarray,
        actions_mask: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        policy = vmap_actor_apply(params.actor_params, next_obs, actions_mask)
        key, subkey = jax.random.split(key)
        raw_action, logprob = policy.sample_and_log_prob(seed=subkey)

        value = vmap_critic_apply(params.critic_params, next_obs)
        return next_obs, raw_action, raw_action, logprob, value.squeeze(), key

    # put data in the last index
    episode_returns = np.zeros((cfg.arch.num_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((cfg.arch.num_envs,), dtype=np.float32)
    episode_lengths = np.zeros((cfg.arch.num_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((cfg.arch.num_envs,), dtype=np.float32)

    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    rollout_queue_put_time = deque(maxlen=10)
    actor_policy_version = 0
    next_obs, infos = envs.reset()
    next_done = jnp.zeros(cfg.arch.num_envs, dtype=jax.numpy.bool_)

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        return jax.tree_map(
            lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage
        )

    for update in range(1, cfg.system.num_updates + 2):
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        env_send_time = 0

        params_queue_get_time_start = time.time()
        if cfg.arch.concurrency:
            if update != 2:
                params = params_queue.get()
                params.network_params["params"]["Dense_0"]["kernel"].block_until_ready()
                actor_policy_version += 1
        else:
            params = params_queue.get()
            actor_policy_version += 1
        params_queue_get_time.append(time.time() - params_queue_get_time_start)
        rollout_time_start = time.time()
        storage = []
        for _ in range(0, cfg.system.rollout_length):
            cached_next_obs = next_obs
            cached_next_done = next_done
            cached_infos = infos
            global_step += (
                len(next_done)
                * cfg.arch.n_threads_per_executor
                * len_executor_device_ids
                * cfg.arch.world_size
            )
            actions_mask = np.stack(cached_infos["actions_mask"])
            inference_time_start = time.time()
            (
                cached_next_obs,
                raw_action,
                action,
                logprob,
                value,
                key,
            ) = get_action_and_value(params, cached_next_obs, actions_mask, key)
            inference_time += time.time() - inference_time_start

            cpu_action = np.array(action)

            env_send_time_start = time.time()
            next_obs, next_reward, next_done, _, infos = envs.step(cpu_action)
            if cfg.system.use_team_reward:
                next_rewards, next_dones, cached_next_dones = jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x, cfg.system.num_agents).reshape(cfg.arch.num_envs, -1),
                    (next_reward, next_done, cached_next_done),
                )

            else:
                next_dones, cached_next_dones = jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x, cfg.system.num_agents).reshape(cfg.arch.num_envs, -1),
                    (next_done, cached_next_done),
                )
                next_rewards = next_reward

            # Accumulate components of reward for logger
            # for cmp in reward_components.keys():
            #     reward_components[cmp] += np.mean([info[cmp] for info in infos])

            # TODO (Ruan): Check what envpool puts in info and make suitable wrapper.
            # env_id = info["env_id"]
            # Hack to make it work like envpool
            env_id = np.arange(cfg.arch.num_envs)
            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()

            truncated = np.zeros_like(next_done)
            truncateds = np.zeros_like(next_dones)
            storage.append(
                Transition(
                    obs=cached_next_obs,
                    dones=cached_next_dones,
                    # only store the raw actions for training.
                    actions=raw_action,
                    logprobs=logprob,
                    values=value,
                    env_ids=env_id,
                    rewards=next_rewards,
                    truncations=truncateds,
                    terminations=next_dones,
                    actions_mask=actions_mask,
                )
            )
            if cfg.system.use_team_reward:
                episode_returns[
                    env_id
                ] += next_reward  # Not sure if this should be current previous reward
            else:
                episode_returns[env_id] += np.mean(next_reward)
            returned_episode_returns[env_id] = np.where(
                next_done + truncated,  # not sure if prev or current done
                episode_returns[env_id],
                returned_episode_returns[env_id],
            )
            episode_returns[env_id] *= (1 - next_done) * (1 - truncated)
            episode_lengths[env_id] += 1
            returned_episode_lengths[env_id] = np.where(
                next_done + truncated,
                episode_lengths[env_id],
                returned_episode_lengths[env_id],
            )
            episode_lengths[env_id] *= (1 - next_done) * (1 - truncated)
            storage_time += time.time() - storage_time_start
        rollout_time.append(time.time() - rollout_time_start)

        partitioned_storage = prepare_data(storage)
        sharded_storage = Transition(
            *list(
                map(
                    lambda x: jax.device_put_sharded(x, devices=learner_devices),
                    partitioned_storage,
                )
            )
        )
        # next_obs, next_done are still in the host
        sharded_next_obs = jax.device_put_sharded(
            np.split(next_obs, len(learner_devices)), devices=learner_devices
        )
        sharded_next_done = jax.device_put_sharded(
            np.split(next_dones, len(learner_devices)), devices=learner_devices
        )
        payload = (
            global_step,
            actor_policy_version,
            update,
            sharded_storage,
            sharded_next_obs,
            sharded_next_done,
            np.mean(params_queue_get_time),
            device_thread_id,
        )
        rollout_queue_put_time_start = time.time()
        rollout_queue.put(payload)
        rollout_queue_put_time.append(time.time() - rollout_queue_put_time_start)

        if (update % cfg.arch.log_frequency == 0) or (cfg.system.num_updates + 1 == update):
            # TODO:  make logger
            log_fn(
                log_type={"Executor": {"device_thread_id": device_thread_id}},
                t_env=global_step,
                metrics_to_log={
                    "episode_info": {
                        "episode_return": returned_episode_returns,
                        "episode_length": returned_episode_lengths,
                    },
                    "speed_info": {
                        "sps": int(global_step / (time.time() - start_time)),
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


def get_trainer(cfg, optim_updates_fn):
    # TODO: get this from main function
    actor_network = Actor(action_dim=cfg.system.num_actions)
    critic_network = Critic()
    vmap_actor_apply = jax.vmap(actor_network.apply, in_axes=(None, 1, 1), out_axes=(1))
    vmap_critic_apply = jax.vmap(critic_network.apply, in_axes=(None, 1), out_axes=(1))
    update_actor_optim, update_critic_optim = optim_updates_fn

    @jax.jit
    def get_logprob_entropy(
        actor_params: flax.core.FrozenDict,
        obs: np.ndarray,
        actions_mask: np.ndarray,
        actions: np.ndarray,
    ):
        policy = vmap_actor_apply(actor_params, obs, actions_mask)

        # Can just pass in raw actions here.
        # No longer need to undo clipping since we are only
        # storing the raw action produced by the clipped policy.
        logprob = policy.log_prob(actions)
        entropy = policy.entropy()
        return logprob, entropy

    @jax.jit
    def get_value(
        critic_params: flax.core.FrozenDict,
        obs: np.ndarray,
    ):
        value = vmap_critic_apply(critic_params, obs)
        return value

    def compute_gae_once(carry, inp, gamma, gae_lambda):
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

    @jax.jit
    def compute_gae(
        agents_state: LearnerState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Transition,
    ):
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

    def policy_loss(actor_params, obs, actions_mask, actions, behavior_logprobs, advantages):
        newlogprob, entropy = get_logprob_entropy(actor_params, obs, actions_mask, actions)
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

    def critic_loss(critic_params, obs, target_values):
        newvalue = get_value(critic_params, obs)
        # Value loss
        v_loss = 0.5 * ((newvalue - target_values) ** 2).mean()
        loss = v_loss * cfg.system.vf_coef
        return loss, (v_loss)

    @jax.jit
    def single_device_update(
        agents_state: LearnerState,
        sharded_storages: List,
        sharded_next_obs: List,
        sharded_next_done: List,
        key: jax.random.PRNGKey,
    ):
        storage = jax.tree_map(lambda *x: jnp.hstack(x), *sharded_storages)
        next_obs = jnp.concatenate(sharded_next_obs)
        next_done = jnp.concatenate(sharded_next_done)
        policy_loss_grad_fn = jax.value_and_grad(policy_loss, has_aux=True)
        critic_loss_grad_fn = jax.value_and_grad(critic_loss, has_aux=True)
        local_advantages, target_values = compute_gae(agents_state, next_obs, next_done, storage)
        # NOTE: advantage normalization at the mini-batch level across devices
        # TODO(Ruan + Omayma): Double check this. But should be correct.
        if cfg.system.norm_adv:
            all_advantages = jax.lax.all_gather(local_advantages, axis_name="local_devices")
            # all_advantages shape: (num_devices, rollout_length, num_envs, num_agents)

            advantages = jnp.concatenate(all_advantages, axis=1)
            # advantages shape: (rollout_length, num_envs * num_devices, num_agents)

            # Normalize advantages across environments and agents
            mean_advantages = advantages.mean((1, 2), keepdims=True)
            std_advantages = advantages.std((1, 2), keepdims=True)
            advantages = (advantages - mean_advantages) / (std_advantages + 1e-8)

            # Split advantages across devices
            split_advantages = jnp.split(advantages, all_advantages.shape[0], axis=1)
            local_advantages = split_advantages[jax.process_index()]
            # local_advantages shape: (rollout_length, num_envs, num_agents)

        def update_epoch(carry, _):
            agents_state, key = carry
            key, subkey = jax.random.split(key)

            def flatten(x):
                return x.reshape((-1,) + x.shape[2:])

            # taken from: https://github.com/google/brax/blob/main/brax/training/agents/ppo/train.py
            def convert_data(x: jnp.ndarray):
                x = jax.random.permutation(subkey, x)
                x = jnp.reshape(
                    x,
                    (
                        cfg.system.num_minibatches * cfg.system.gradient_accumulation_steps,
                        -1,
                    )
                    + x.shape[1:],
                )
                return x

            # TODO(Ruan): Need to keep the agent dimension here.
            # Storage is (rollout_len, env, *shape) -> (rollout_len * env, *shape)
            # Advantages + values is (rollout_len, env, agent) -> (rollout_len * env, agent)

            # Shuffled storage is (rollout_len * env, *shape) -> (num_minibatches, minibatch_size, *shape)
            # Shuffled advantages + values is (rollout_len * env, ) -> (num_minibatches, minibatch_size)
            def flatten_and_shuffle(storage, local_advantages, target_values):
                flatten_storage = jax.tree_map(flatten, storage)
                flatten_advantages = flatten(local_advantages)
                flatten_target_values = flatten(target_values)
                shuffled_storage = jax.tree_map(convert_data, flatten_storage)
                shuffled_advantages = convert_data(flatten_advantages)
                shuffled_target_values = convert_data(flatten_target_values)
                return (shuffled_storage, shuffled_advantages, shuffled_target_values)

            vmap_flatten_and_shuffle = jax.vmap(
                flatten_and_shuffle,
                in_axes=(
                    Transition(
                        obs=2,
                        dones=2,
                        actions=2,
                        logprobs=2,
                        values=2,
                        env_ids=None,
                        rewards=2,
                        truncations=2,
                        terminations=2,
                        actions_mask=2,
                    ),
                    2,
                    2,
                ),
                out_axes=(
                    Transition(
                        obs=2,
                        dones=2,
                        actions=2,
                        logprobs=2,
                        values=2,
                        env_ids=None,
                        rewards=2,
                        truncations=2,
                        terminations=2,
                        actions_mask=2,
                    ),
                    2,
                    2,
                ),
            )

            (
                shuffled_storage,
                shuffled_advantages,
                shuffled_target_values,
            ) = vmap_flatten_and_shuffle(storage, local_advantages, target_values)

            def update_minibatch(agents_state, minibatch):
                (
                    mb_obs,
                    mb_actions,
                    mb_behavior_logprobs,
                    mb_advantages,
                    mb_target_values,
                    mb_actions_mask,
                ) = minibatch
                (p_loss, (pg_loss, entropy_loss, approx_kl),), actor_grads = policy_loss_grad_fn(
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
                actor_grads = jax.lax.pmean(actor_grads, axis_name="local_devices")
                critic_grads = jax.lax.pmean(critic_grads, axis_name="local_devices")

                loss = p_loss + c_loss
                # UPDATE ACTOR PARAMS AND OPTIMISER STATE
                actor_updates, actor_new_opt_state = update_actor_optim(
                    actor_grads, agents_state.opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(
                    agents_state.params.actor_params, actor_updates
                )

                # UPDATE CRITIC PARAMS AND OPTIMISER STATE
                critic_updates, critic_new_opt_state = update_critic_optim(
                    critic_grads, agents_state.opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(
                    agents_state.params.critic_params, critic_updates
                )

                # PACK NEW PARAMS AND OPTIMISER STATE
                new_params = Params(actor_new_params, critic_new_params)
                new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)

                agents_state = LearnerState(params=new_params, opt_states=new_opt_state)
                return agents_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl)

            agents_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl,) = jax.lax.scan(
                update_minibatch,
                agents_state,
                (
                    shuffled_storage.obs,
                    shuffled_storage.actions,
                    shuffled_storage.logprobs,
                    shuffled_advantages,
                    shuffled_target_values,
                    shuffled_storage.actions_mask,
                ),
            )
            return (agents_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl)

        (agents_state, key), (
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
        ) = jax.lax.scan(update_epoch, (agents_state, key), (), length=cfg.system.ppo_epochs)
        loss = jax.lax.pmean(loss, axis_name="local_devices").mean()
        pg_loss = jax.lax.pmean(pg_loss, axis_name="local_devices").mean()
        v_loss = jax.lax.pmean(v_loss, axis_name="local_devices").mean()
        entropy_loss = jax.lax.pmean(entropy_loss, axis_name="local_devices").mean()
        approx_kl = jax.lax.pmean(approx_kl, axis_name="local_devices").mean()
        loss_info = {
            "total_loss": loss,
            "loss_actor": pg_loss,
            "value_loss": v_loss,
            "entropy": entropy_loss,
            "approx_kl": approx_kl,
        }
        return agents_state, key, loss_info

    return single_device_update


def run_experiment(cfg: DictConfig) -> None:
    """Run experiment."""
    # Logger setup
    cfg_dict: Dict = OmegaConf.to_container(cfg, resolve=True)
    log = logger_setup(cfg_dict)

    # Create eval_envs and dummy_envs to get observation and action spaces
    eval_env = make_env(
        num_envs=cfg.arch.num_eval_episodes,
        config=cfg,
    )()
    dummy_envs = make_env(
        num_envs=cfg.arch.num_envs,
        config=cfg,
    )()

    # Config setup
    cfg.system.local_batch_size = int(
        cfg.arch.num_envs
        * cfg.system.rollout_length
        * cfg.arch.n_threads_per_executor
        * len(cfg.arch.executor_device_ids)
    )
    cfg.system.local_minibatch_size = int(cfg.system.local_batch_size // cfg.system.num_minibatches)
    assert (
        cfg.arch.num_envs % len(cfg.arch.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(cfg.arch.num_envs / len(cfg.arch.learner_device_ids))
        * cfg.arch.n_threads_per_executor
        % cfg.system.num_minibatches
        == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
    cfg.arch.world_size = jax.process_count()
    cfg.arch.local_rank = jax.process_index()
    cfg.system.batch_size = cfg.system.local_batch_size * cfg.arch.world_size
    cfg.system.minibatch_size = cfg.system.local_minibatch_size * cfg.arch.world_size
    cfg.system.total_timesteps = cfg.system.num_updates * (
        cfg.system.local_batch_size * cfg.arch.world_size
    )
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in cfg.arch.learner_device_ids]
    actor_devices = [local_devices[d_id] for d_id in cfg.arch.executor_device_ids]
    global_learner_devices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(cfg.arch.world_size)
        for d_id in cfg.arch.learner_device_ids
    ]
    cfg.system.global_learner_devices = [str(item) for item in global_learner_devices]
    cfg.system.actor_devices = [str(item) for item in actor_devices]
    cfg.system.learner_devices = [str(item) for item in learner_devices]
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
    single_device_update = get_trainer(cfg, (actor_optim.update, critic_optim.update))
    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_devices,
    )

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
                    local_devices[d_id],
                    log,
                ),
            ).start()

    rollout_queue_get_time: deque = deque(maxlen=10)
    data_transfer_time: deque = deque(maxlen=10)
    trainer_update_number = 0
    log_frequency = cfg.system.num_updates // cfg.arch.num_evaluation
    while True:
        trainer_update_number += 1
        rollout_queue_get_time_start = time.time()
        sharded_storages = []
        sharded_next_obss = []
        sharded_next_dones = []

        # Loop through each executor device
        for d_idx, d_id in enumerate(cfg.arch.executor_device_ids):
            # Loop through each executor thread
            for thread_id in range(cfg.arch.n_threads_per_executor):
                # Get data from rollout queue
                (
                    global_step,
                    actor_policy_version,
                    update,
                    sharded_storage,
                    sharded_next_obs,
                    sharded_next_done,
                    avg_params_queue_get_time,
                    device_thread_id,
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
                t_env=global_step,
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
            evaluation(
                envs=eval_env,
                params=unreplicated_params,
                vmap_actor_apply=vmap_actor_apply,
                key=eval_key,
                cfg=cfg,
                log_fn=log,
                t_env=global_step,
            )

        # Check if training is finished
        if trainer_update_number >= cfg.system.num_updates:
            eval_key, _ = jax.random.split(eval_key)
            evaluation(
                envs=eval_env,
                params=unreplicated_params,
                vmap_actor_apply=vmap_actor_apply,
                key=eval_key,
                cfg=cfg,
                log_fn=log,
                t_env=global_step,
            )
            break


@hydra.main(config_path="../configs", config_name="default_ff_ippo.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> None:
    """Experiment entry point."""

    # Run experiment.
    run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}IPPO experiment completed{Style.RESET_ALL}")


if __name__ == "__main__":
    configure_computation_environment()
    hydra_entry_point()
