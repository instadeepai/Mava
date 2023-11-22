import functools
import os
import queue
import random
import threading
import time
import uuid
from collections import deque
from functools import partial
from multiprocessing import set_start_method
from types import SimpleNamespace
from typing import List, NamedTuple, Sequence

import chex
import distrax
import flax
import flax.linen as nn
import gym
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import DictConfig, OmegaConf
from optax._src.base import OptState
from rich.pretty import pprint

from mava.logger import Logger
from mava.wrappers.gym_wrapper import make_env as make_env_single

os.environ["JAX_USE_PJRT_C_API_ON_TPU"] = ""


# Fix weird OOM https://github.com/google/jax/discussions/6332#discussioncomment-1279991
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.6"
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false " "intra_op_parallelism_threads=1"
# Fix CUDNN non-determinisim; https://github.com/google/jax/issues/4823#issuecomment-952835771
os.environ["TF_XLA_FLAGS"] = "--xla_gpu_autotune_level=2 --xla_gpu_deterministic_reductions"
os.environ["TF_CUDNN DETERMINISTIC"] = "1"


def make_env(
    env_id: str,
    seed: int,
    num_envs: int,
    use_team_reward: bool = True,
    async_envs: bool = False,
    training_mode: bool = True,
):
    def thunk():
        if async_envs:
            envs = gym.vector.AsyncVectorEnv(
                [
                    make_env_single(
                        task_name="rware-tiny-2ag-v1",
                        team_reward=use_team_reward,
                    )
                    for i in range(num_envs)
                ]
            )
        else:
            envs = gym.vector.SyncVectorEnv(
                [
                    make_env_single(
                        task_name="rware-tiny-2ag-v1",
                        team_reward=use_team_reward,
                    )
                    for i in range(num_envs)
                ]
            )
        envs.num_envs = num_envs
        envs.is_vector_env = True
        return envs

    return thunk


class Actor(nn.Module):
    """Actor Network."""

    action_dim: Sequence[int]

    @nn.compact
    def __call__(self, observation: chex.Array) -> distrax.Categorical:
        """Forward pass."""

        actor_output = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            observation
        )
        actor_output = nn.relu(actor_output)
        actor_output = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(
            actor_output
        )
        actor_output = nn.relu(actor_output)
        actor_logits = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_output)

        return actor_logits


class Critic(nn.Module):
    """Critic Network."""

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
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


class Params(NamedTuple):
    """Parameters of an actor critic network."""

    actor_params: FrozenDict
    critic_params: FrozenDict


class OptStates(NamedTuple):
    """OptStates of actor critic learner."""

    actor_opt_state: OptState
    critic_opt_state: OptState


class TrainerState(NamedTuple):
    """Parameters of an actor critic network."""

    params: Params
    opt_states: OptStates


class Transition(NamedTuple):
    obs: list
    dones: list
    actions: list
    logprobs: list
    values: list
    env_ids: list
    rewards: list
    truncations: list
    terminations: list


def rollout(
    key: jax.random.PRNGKey,
    cfg,
    rollout_queue,
    params_queue: queue.Queue,
    writer,
    learner_devices,
    device_thread_id,
    actor_device,
):
    envs = make_env(
        env_id=cfg.system.env_id,
        seed=cfg.system.seed + jax.process_index() + device_thread_id,
        num_envs=cfg.arch.local_n_envs,
        use_team_reward=cfg.system.use_team_reward,
        async_envs=cfg.arch.async_envs,
        training_mode=True,
    )()
    len_actor_device_ids = len(cfg.arch.actor_device_ids)
    global_step = 0
    start_time = time.time()

    @jax.jit
    def get_action_and_value(  # TODO: Use the power_tools action methods?
        params: flax.core.FrozenDict,
        next_obs: np.ndarray,
        key: jax.random.PRNGKey,
    ):
        next_obs = jnp.array(next_obs)
        actor_network = Actor(action_dim=cfg.system.num_actions)
        vmap_actor_apply = jax.vmap(actor_network.apply, in_axes=(None, 1), out_axes=(1))
        logits = vmap_actor_apply(params.actor_params, next_obs)

        policy = distrax.Categorical(logits=logits)

        key, subkey = jax.random.split(key)
        raw_action, logprob = policy.sample_and_log_prob(seed=subkey)

        critic_network = Critic()
        vmap_critic_apply = jax.vmap(critic_network.apply, in_axes=(None, 1), out_axes=(1))
        value = vmap_critic_apply(params.critic_params, next_obs)
        return next_obs, raw_action, raw_action, logprob, value.squeeze(), key

    # put data in the last index
    episode_returns = np.zeros((cfg.arch.local_n_envs,), dtype=np.float32)
    returned_episode_returns = np.zeros((cfg.arch.local_n_envs,), dtype=np.float32)
    episode_lengths = np.zeros((cfg.arch.local_n_envs,), dtype=np.float32)
    returned_episode_lengths = np.zeros((cfg.arch.local_n_envs,), dtype=np.float32)

    params_queue_get_time = deque(maxlen=10)
    rollout_time = deque(maxlen=10)
    rollout_queue_put_time = deque(maxlen=10)
    actor_policy_version = 0
    next_obs = envs.reset()
    # Citylearn had a single done. Rware has a done per agent.
    next_done = jnp.zeros((cfg.arch.local_n_envs, cfg.system.num_agents), dtype=jax.numpy.bool_)

    @jax.jit
    def prepare_data(storage: List[Transition]) -> Transition:
        return jax.tree_map(
            lambda *xs: jnp.split(jnp.stack(xs), len(learner_devices), axis=1), *storage
        )

    for update in range(1, cfg.system.num_updates + 2):
        update_time_start = time.time()
        env_recv_time = 0
        inference_time = 0
        storage_time = 0
        d2h_time = 0
        env_send_time = 0
        # NOTE: `update != 2` is actually IMPORTANT — it allows us to start running policy collection
        # concurrently with the learning process. It also ensures the actor's policy version is only 1 step
        # behind the learner's policy version
        params_queue_get_time_start = time.time()
        if cfg.arch.concurrency:
            if update != 2:
                params = params_queue.get()
                # NOTE: block here is important because otherwise this thread will call
                # the jitted `get_action_and_value` function that hangs until the params are ready.
                # This blocks the `get_action_and_value` function in other actor threads.
                # See https://excalidraw.com/#json=hSooeQL707gE5SWY8wOSS,GeaN1eb2r24PPi75a3n14Q for a visual explanation.
                params.network_params["params"]["Dense_0"][
                    "kernel"
                ].block_until_ready()  # TODO: check if params.block_until_ready() is enough
                actor_policy_version += 1
        else:
            params = params_queue.get()
            actor_policy_version += 1
        params_queue_get_time.append(time.time() - params_queue_get_time_start)
        rollout_time_start = time.time()
        storage = []
        for _ in range(0, cfg.system.num_steps):
            cached_next_obs = next_obs
            cached_next_done = next_done
            global_step += (
                len(next_done)
                * cfg.arch.n_threads_per_actor
                * len_actor_device_ids
                * cfg.arch.world_size
            )
            inference_time_start = time.time()
            (
                cached_next_obs,
                raw_action,
                action,
                logprob,
                value,
                key,
            ) = get_action_and_value(params, cached_next_obs, key)
            inference_time += time.time() - inference_time_start

            d2h_time_start = time.time()
            cpu_action = np.array(action)
            d2h_time += time.time() - d2h_time_start

            env_send_time_start = time.time()
            next_obs, next_reward, next_done, infos = envs.step(cpu_action)
            if cfg.system.use_team_reward:
                next_rewards = jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x, cfg.system.num_agents).reshape(
                        cfg.arch.local_n_envs, -1
                    ),
                    next_reward,
                )

            next_dones, cached_next_dones = next_done, cached_next_done

            # Accumulate components of reward for logger
            # for cmp in reward_components.keys():
            #     reward_components[cmp] += np.mean([info[cmp] for info in infos])

            # TODO (Ruan): Check what envpool puts in info and make suitable wrapper.
            # env_id = info["env_id"]
            # Hack to make it work like envpool
            env_id = np.arange(cfg.arch.local_n_envs)
            env_send_time += time.time() - env_send_time_start
            storage_time_start = time.time()

            # info["TimeLimit.truncated"] has a bug https://github.com/sail-sg/envpool/issues/239
            # so we use our own truncated flag

            # truncated = info["elapsed_step"] >= envs.spec.config.max_episode_steps
            # Vanilla gym envs don't truncate
            # Should always be False
            truncated = np.zeros_like(next_done)
            truncateds = np.zeros_like(next_dones)
            storage.append(
                Transition(
                    obs=cached_next_obs,
                    dones=cached_next_dones,
                    actions=raw_action,  # only store the raw actions for training.
                    logprobs=logprob,
                    values=value,
                    env_ids=env_id,
                    rewards=next_rewards,
                    truncations=truncateds,
                    terminations=next_dones,
                )
            )
            if cfg.system.use_team_reward:
                episode_returns[
                    env_id
                ] += next_reward  # Not sure if this should be current previous reward
            else:
                episode_returns[env_id] += np.mean(next_reward)
            returned_episode_returns[env_id] = np.where(
                next_done
                + truncated,  # TODO: Hack to get things working  # not sure if prev or current done
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

        avg_episodic_return = np.mean(returned_episode_returns)
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
            if device_thread_id == 0:
                print(
                    f"global_step={global_step}, avg_episodic_return={avg_episodic_return}, rollout_time={np.mean(rollout_time)}"
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
            writer.log_stat("stats/rollout_time", np.mean(rollout_time), global_step)
            writer.log_stat("charts/avg_episodic_return", avg_episodic_return, global_step)
            writer.log_stat(
                "charts/avg_episodic_length",
                np.mean(returned_episode_lengths),
                global_step,
            )
            writer.log_stat(
                "stats/params_queue_get_time",
                np.mean(params_queue_get_time),
                global_step,
            )
            writer.log_stat("stats/env_recv_time", env_recv_time, global_step)
            writer.log_stat("stats/inference_time", inference_time, global_step)
            writer.log_stat("stats/storage_time", storage_time, global_step)
            writer.log_stat("stats/d2h_time", d2h_time, global_step)
            writer.log_stat("stats/env_send_time", env_send_time, global_step)
            writer.log_stat(
                "stats/rollout_queue_put_time",
                np.mean(rollout_queue_put_time),
                global_step,
            )
            writer.log_stat(
                "charts/SPS", int(global_step / (time.time() - start_time)), global_step
            )
            writer.log_stat(
                "charts/SPS_update",
                int(
                    cfg.arch.local_n_envs
                    * cfg.system.num_steps
                    * len_actor_device_ids
                    * cfg.arch.n_threads_per_actor
                    * cfg.arch.world_size
                    / (time.time() - update_time_start)
                ),
                global_step,
            )

            # for reward_name, reward_val in reward_components.items():
            #     writer.log_stat(f"reward/{reward_name}", reward_val, global_step)
            #     reward_components[reward_name] = 0.0


@hydra.main(version_base=None, config_path="../../configs", config_name="default_ff_ippo_sebulba")
def run(cfg: DictConfig):
    cfg.system.local_batch_size = int(
        cfg.arch.local_n_envs
        * cfg.system.num_steps
        * cfg.arch.n_threads_per_actor
        * len(cfg.arch.actor_device_ids)
    )
    eval_env = make_env(
        env_id=None,
        seed=0,
        num_envs=cfg.arch.n_evals_envs,
        use_team_reward=cfg.system.use_team_reward,
        async_envs=True,
        training_mode=False,
    )()
    cfg.system.local_minibatch_size = int(cfg.system.local_batch_size // cfg.system.num_minibatches)
    assert (
        cfg.arch.local_n_envs % len(cfg.arch.learner_device_ids) == 0
    ), "local_num_envs must be divisible by len(learner_device_ids)"
    assert (
        int(cfg.arch.local_n_envs / len(cfg.arch.learner_device_ids))
        * cfg.arch.n_threads_per_actor
        % cfg.system.num_minibatches
        == 0
    ), "int(local_num_envs / len(learner_device_ids)) must be divisible by num_minibatches"
    if cfg.arch.distributed:
        jax.distributed.initialize(
            local_device_ids=range(
                len(cfg.arch.learner_device_ids) + len(cfg.arch.actor_device_ids)
            ),
        )
        print(list(range(len(cfg.arch.learner_device_ids) + len(cfg.arch.actor_device_ids))))

    cfg.arch.world_size = jax.process_count()
    cfg.arch.local_rank = jax.process_index()
    cfg.system.num_envs = (
        cfg.arch.local_n_envs
        * cfg.arch.world_size
        * cfg.arch.n_threads_per_actor
        * len(cfg.arch.actor_device_ids)
    )
    cfg.system.batch_size = cfg.system.local_batch_size * cfg.arch.world_size
    cfg.system.minibatch_size = cfg.system.local_minibatch_size * cfg.arch.world_size
    cfg.system.num_updates = cfg.system.total_timesteps // (
        cfg.system.local_batch_size * cfg.arch.world_size
    )
    local_devices = jax.local_devices()
    global_devices = jax.devices()
    learner_devices = [local_devices[d_id] for d_id in cfg.arch.learner_device_ids]
    actor_devices = [local_devices[d_id] for d_id in cfg.arch.actor_device_ids]
    global_learner_devices = [
        global_devices[d_id + process_index * len(local_devices)]
        for process_index in range(cfg.arch.world_size)
        for d_id in cfg.arch.learner_device_ids
    ]
    print("global_learner_decices", global_learner_devices)
    cfg.system.global_learner_devices = [str(item) for item in global_learner_devices]
    cfg.system.actor_devices = [str(item) for item in actor_devices]
    cfg.system.learner_devices = [str(item) for item in learner_devices]

    run_name = f"{cfg.system.env_id}__{cfg.logger.system_name}__{cfg.system.seed}__{uuid.uuid4()}"

    # seeding
    random.seed(cfg.system.seed)
    np.random.seed(cfg.system.seed)
    key = jax.random.PRNGKey(cfg.system.seed)
    key, network_key, actor_key, critic_key = jax.random.split(key, 4)
    learner_keys = jax.device_put_replicated(key, learner_devices)

    # env setup
    envs = make_env(
        env_id=cfg.system.env_id,
        seed=cfg.system.seed,
        num_envs=cfg.arch.local_n_envs,
        use_team_reward=cfg.system.use_team_reward,
        training_mode=True,
    )()
    cfg.system.num_agents = envs.single_observation_space.shape[0]  # must be 3
    cfg.system.num_actions = int(envs.single_action_space.nvec[0])  # can definitely be done better
    cfg.system.single_obs_dim = envs.single_observation_space.shape[1]  # must be 32
    cfg_dict = OmegaConf.to_container(cfg, resolve=True)
    pprint(cfg_dict)

    # TODO: Set correctly in the config.

    writer = Logger(cfg_dict)
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s"
    #     % ("\n".join([f"|{key}|{value}|" for key, value in cfg_dict.items()])),
    # )

    actor = Actor(action_dim=cfg.system.num_actions)
    # vmapped_actor = jax.vmap(actor.apply, in_axes=(None, 1), out_axes=(1))
    # select_actions = jax.jit(functools.partial(select_actions_ppo_ff, vmapped_actor))

    critic = Critic()

    # Create checkpointer if necessary
    if cfg.arch.save_model:
        raise NotImplementedError

    # Initialise params.
    actor_params = actor.init(
        network_key,
        np.array([envs.single_observation_space.sample()[0]]),  # should be (1, 30)
    )
    critic_params = critic.init(
        network_key,
        np.array([envs.single_observation_space.sample()[0]]),  # should be (1, 30)
    )

    def critic_linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches) gradient updates
        frac = (
            1.0
            - (count // (cfg.system.num_minibatches * cfg.system.update_epochs))
            / cfg.system.num_updates
        )
        return cfg.system.critic_lr * frac

    def actor_linear_schedule(count):
        # anneal learning rate linearly after one training iteration which contains
        # (args.num_minibatches) gradient updates
        frac = (
            1.0
            - (count // (cfg.system.num_minibatches * cfg.system.update_epochs))
            / cfg.system.num_updates
        )
        return cfg.system.actor_lr * frac

    actor_optim = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(cfg.system.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=actor_linear_schedule
                if cfg.system.anneal_lr
                else cfg.system.actor_lr,
                eps=1e-5,
            ),
        ),
        every_k_schedule=cfg.system.gradient_accumulation_steps,
    )
    actor_opt_state = actor_optim.init(actor_params)

    critic_optim = optax.MultiSteps(
        optax.chain(
            optax.clip_by_global_norm(cfg.system.max_grad_norm),
            optax.inject_hyperparams(optax.adam)(
                learning_rate=critic_linear_schedule
                if cfg.system.anneal_lr
                else cfg.system.critic_lr,
                eps=1e-5,
            ),
        ),
        every_k_schedule=cfg.system.gradient_accumulation_steps,
    )

    critic_opt_state = critic_optim.init(critic_params)

    agent_state = TrainerState(
        params=Params(
            actor_params=actor_params,
            critic_params=critic_params,
        ),
        opt_states=OptStates(
            actor_opt_state=actor_opt_state,
            critic_opt_state=critic_opt_state,
        ),
    )

    agent_state = flax.jax_utils.replicate(agent_state, devices=learner_devices)
    print(
        actor.tabulate(
            actor_key,
            actor.apply(actor_params, np.array([envs.single_observation_space.sample()])),
        )
    )
    print(
        critic.tabulate(
            critic_key,
            critic.apply(critic_params, np.array([envs.single_observation_space.sample()])),
        )
    )

    @jax.jit
    def get_logprob_entropy(
        actor_params: flax.core.FrozenDict,
        obs: np.ndarray,
        actions: np.ndarray,
    ):
        actor_network = Actor(action_dim=cfg.system.num_actions)
        vmap_actor_apply = jax.vmap(actor_network.apply, in_axes=(None, 1), out_axes=(1))
        logits = vmap_actor_apply(actor_params, obs)
        policy = distrax.Categorical(logits=logits)

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
        critic_network = Critic()
        vmap_critic_apply = jax.vmap(critic_network.apply, in_axes=(None, 1), out_axes=(1))
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
        agent_state: TrainState,
        next_obs: np.ndarray,
        next_done: np.ndarray,
        storage: Transition,
    ):
        critic = Critic()
        single_critic_apply = critic.apply
        vmap_critic_apply = jax.vmap(single_critic_apply, in_axes=(None, 1), out_axes=(1))
        next_value = vmap_critic_apply(
            agent_state.params.critic_params,
            next_obs,
        )

        advantages = jnp.zeros(
            (
                cfg.arch.local_n_envs
                * cfg.arch.n_threads_per_actor
                * len(cfg.arch.actor_device_ids)
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

    def policy_loss(actor_params, obs, actions, behavior_logprobs, advantages):
        newlogprob, entropy = get_logprob_entropy(actor_params, obs, actions)
        logratio = newlogprob - behavior_logprobs
        ratio = jnp.exp(logratio)
        approx_kl = ((ratio - 1) - logratio).mean()

        # Policy loss
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * jnp.clip(ratio, 1 - cfg.system.clip_coef, 1 + cfg.system.clip_coef)
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
        agent_state: TrainState,
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
        local_advantages, target_values = compute_gae(agent_state, next_obs, next_done, storage)
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
            agent_state, key = carry
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

            def update_minibatch(agent_state, minibatch):
                (
                    mb_obs,
                    mb_actions,
                    mb_behavior_logprobs,
                    mb_advantages,
                    mb_target_values,
                ) = minibatch
                (
                    p_loss,
                    (pg_loss, entropy_loss, approx_kl),
                ), actor_grads = policy_loss_grad_fn(
                    agent_state.params.actor_params,
                    mb_obs,
                    mb_actions,
                    mb_behavior_logprobs,
                    mb_advantages,
                )
                (c_loss, (v_loss)), critic_grads = critic_loss_grad_fn(
                    agent_state.params.critic_params,
                    mb_obs,
                    mb_target_values,
                )
                actor_grads = jax.lax.pmean(actor_grads, axis_name="local_devices")
                critic_grads = jax.lax.pmean(critic_grads, axis_name="local_devices")

                loss = p_loss + c_loss
                # UPDATE ACTOR PARAMS AND OPTIMISER STATE
                actor_updates, actor_new_opt_state = actor_optim.update(
                    actor_grads, agent_state.opt_states.actor_opt_state
                )
                actor_new_params = optax.apply_updates(
                    agent_state.params.actor_params, actor_updates
                )

                # UPDATE CRITIC PARAMS AND OPTIMISER STATE
                critic_updates, critic_new_opt_state = critic_optim.update(
                    critic_grads, agent_state.opt_states.critic_opt_state
                )
                critic_new_params = optax.apply_updates(
                    agent_state.params.critic_params, critic_updates
                )

                # PACK NEW PARAMS AND OPTIMISER STATE
                new_params = Params(actor_new_params, critic_new_params)
                new_opt_state = OptStates(actor_new_opt_state, critic_new_opt_state)

                agent_state = TrainerState(params=new_params, opt_states=new_opt_state)
                return agent_state, (loss, pg_loss, v_loss, entropy_loss, approx_kl)

            agent_state, (
                loss,
                pg_loss,
                v_loss,
                entropy_loss,
                approx_kl,
            ) = jax.lax.scan(
                update_minibatch,
                agent_state,
                (
                    shuffled_storage.obs,
                    shuffled_storage.actions,
                    shuffled_storage.logprobs,
                    shuffled_advantages,
                    shuffled_target_values,
                ),
            )
            return (agent_state, key), (loss, pg_loss, v_loss, entropy_loss, approx_kl)

        (agent_state, key), (
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
        ) = jax.lax.scan(update_epoch, (agent_state, key), (), length=cfg.system.update_epochs)
        loss = jax.lax.pmean(loss, axis_name="local_devices").mean()
        pg_loss = jax.lax.pmean(pg_loss, axis_name="local_devices").mean()
        v_loss = jax.lax.pmean(v_loss, axis_name="local_devices").mean()
        entropy_loss = jax.lax.pmean(entropy_loss, axis_name="local_devices").mean()
        approx_kl = jax.lax.pmean(approx_kl, axis_name="local_devices").mean()
        return agent_state, loss, pg_loss, v_loss, entropy_loss, approx_kl, key

    multi_device_update = jax.pmap(
        single_device_update,
        axis_name="local_devices",
        devices=global_learner_devices,
    )

    params_queues = []
    rollout_queues = []
    dummy_writer = SimpleNamespace()
    dummy_writer.add_scalar = lambda x, y, z: None

    unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
    for d_idx, d_id in enumerate(cfg.arch.actor_device_ids):
        device_params = jax.device_put(unreplicated_params, local_devices[d_id])
        for thread_id in range(cfg.arch.n_threads_per_actor):
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
                    writer if d_idx == 0 and thread_id == 0 else dummy_writer,
                    learner_devices,
                    d_idx * cfg.arch.n_threads_per_actor + thread_id,
                    local_devices[d_id],
                ),
            ).start()

    rollout_queue_get_time = deque(maxlen=10)
    data_transfer_time = deque(maxlen=10)
    learner_policy_version = 0
    eval_idx = 0

    while True:
        learner_policy_version += 1
        rollout_queue_get_time_start = time.time()
        sharded_storages = []
        sharded_next_obss = []
        sharded_next_dones = []
        for d_idx, d_id in enumerate(cfg.arch.actor_device_ids):
            for thread_id in range(cfg.arch.n_threads_per_actor):
                (
                    global_step,
                    actor_policy_version,
                    update,
                    sharded_storage,
                    sharded_next_obs,
                    sharded_next_done,
                    avg_params_queue_get_time,
                    device_thread_id,
                ) = rollout_queues[d_idx * cfg.arch.n_threads_per_actor + thread_id].get()
                sharded_storages.append(sharded_storage)
                sharded_next_obss.append(sharded_next_obs)
                sharded_next_dones.append(sharded_next_done)
        rollout_queue_get_time.append(time.time() - rollout_queue_get_time_start)
        training_time_start = time.time()
        (
            agent_state,
            loss,
            pg_loss,
            v_loss,
            entropy_loss,
            approx_kl,
            learner_keys,
        ) = multi_device_update(
            agent_state,
            sharded_storages,
            sharded_next_obss,
            sharded_next_dones,
            learner_keys,
        )
        unreplicated_params = flax.jax_utils.unreplicate(agent_state.params)
        for d_idx, d_id in enumerate(cfg.arch.actor_device_ids):
            device_params = jax.device_put(unreplicated_params, local_devices[d_id])
            for thread_id in range(cfg.arch.n_threads_per_actor):
                params_queues[d_idx * cfg.arch.n_threads_per_actor + thread_id].put(device_params)

        # record rewards for plotting purposes
        if learner_policy_version % cfg.arch.log_frequency == 0:
            writer.log_stat(
                "stats/rollout_queue_get_time",
                np.mean(rollout_queue_get_time),
                global_step,
            )
            writer.log_stat(
                "stats/rollout_params_queue_get_time_diff",
                np.mean(rollout_queue_get_time) - avg_params_queue_get_time,
                global_step,
            )
            writer.log_stat("stats/training_time", time.time() - training_time_start, global_step)
            writer.log_stat("stats/rollout_queue_size", rollout_queues[-1].qsize(), global_step)
            writer.log_stat("stats/params_queue_size", params_queues[-1].qsize(), global_step)
            print(
                global_step,
                f"actor_policy_version={actor_policy_version}, actor_update={update}, learner_policy_version={learner_policy_version}, training time: {time.time() - training_time_start}s",
            )

            writer.log_stat("losses/value_loss", v_loss[-1].item(), global_step)
            writer.log_stat("losses/policy_loss", pg_loss[-1].item(), global_step)
            writer.log_stat("losses/entropy", entropy_loss[-1].item(), global_step)
            writer.log_stat("losses/approx_kl", approx_kl[-1].item(), global_step)
            writer.log_stat("losses/loss", loss[-1].item(), global_step)

            # # Evaluation
            # key, eval_rng = jax.random.split(key)

            # eval_logs = evaluation(
            #     eval_env,
            #     cfg.arch.n_eval_episodes,
            #     unreplicated_params.actor_params,
            #     select_actions,
            #     writer.log_stat,
            #     eval_idx,
            #     eval_rng,
            # )

            # for metric, value in eval_logs.items():
            #     writer.log_stat(f"charts/{metric}", value, global_step)

            # print(f"EVALUATION: Average eval score - {eval_logs['average_score']}")

            # # Save model
            # if cfg.arch.save_model:
            #     checkpointer.save(
            #         global_step,
            #         unreplicated_params.actor_params,
            #         eval_logs,
            #     )
        if learner_policy_version >= cfg.system.num_updates:
            key, eval_rng = jax.random.split(key)
            # eval_logs = evaluation(
            #     eval_env,
            #     cfg.arch.n_eval_episodes,
            #     unreplicated_params.actor_params,
            #     select_actions,
            #     writer.log_stat,
            #     eval_idx + 1,
            #     eval_rng,
            # )

            # for metric, value in eval_logs.items():
            #     writer.log_stat(f"charts/{metric}", value, global_step)

            # print(
            #     f"FINAL EVALUATION: Average eval score - {eval_logs['average_score']}"
            # )

            # # Save model
            # if cfg.arch.save_model:
            #     checkpointer.save(
            #         global_step,
            #         unreplicated_params.actor_params,
            #         eval_logs,
            #     )
            break

    envs.close()
    writer.close()
    return 1.0


if __name__ == "__main__":
    set_start_method("forkserver")
    run()
