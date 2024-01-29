# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, ClassVar, Dict, NamedTuple, Optional, Tuple, Union

import chex
import distrax
import flashbax as fbx
import flax.linen as nn
import gymnasium as gym

# import gymnasium_robotics
import jax
import jax.numpy as jnp

import jaxmarl

# from torch.utils.tensorboard import SummaryWriter
import neptune
import numpy as np
import optax
import tensorboard_logger
import tyro
from chex import PRNGKey
from gymnasium.spaces import Box
from jax import Array
from jax.typing import ArrayLike
from jumanji import specs
from jumanji.env import Environment, State
from jumanji.types import Observation, TimeStep
from jumanji.wrappers import GymObservation, VmapWrapper, Wrapper, jumanji_to_gym_obs

from mava.wrappers import RecordEpisodeMetrics
from mava.wrappers.jaxmarl import JaxMarlWrapper


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 42
    """seed of the experiment"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    # env_id: str = "MPE_simple_spread_v3"
    env_id: str = "halfcheetah_6x1"
    """the environment id of the task"""
    factorization: str = "2x3"
    """how the joints are split up"""
    total_timesteps: int = int(3e8)
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = int(5e3)
    """timestep to start learning"""
    policy_lr: float = 3e-4
    """the learning rate of the policy network optimizer"""
    q_lr: float = 1e-3
    """the learning rate of the Q network network optimizer"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    target_network_frequency: int = 1  # Denis Yarats' implementation delays this by 2.
    """the frequency of updates for the target nerworks"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    n_envs: int = 16
    """number of parallel environments"""


class VmapAutoResetWrapper(Wrapper):
    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        state, timestep = jax.vmap(self._env.reset)(key)
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        # Vmap homogeneous computation (parallelizable).
        state, timestep = jax.vmap(self._env.step)(state, action)
        # Map heterogeneous computation (non-parallelizable).
        state, timestep = jax.lax.map(lambda args: self._maybe_reset(*args), (state, timestep))
        return state, timestep

    def _auto_reset(self, state: State, timestep: TimeStep) -> Tuple[State, TimeStep[Observation]]:
        if not hasattr(state, "key"):
            raise AttributeError(
                "This wrapper assumes that the state has attribute key which is used"
                " as the source of randomness for automatic reset"
            )

        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        key, _ = jax.random.split(state.key)
        state, reset_timestep = self._env.reset(key)

        extras = timestep.extras
        extras["final_observation"] = timestep.observation

        # Replace observation with reset observation.
        timestep = timestep.replace(  # type: ignore
            observation=reset_timestep.observation, extras=extras
        )

        return state, timestep

    def _obs_in_extras(self, timestep: TimeStep[Observation]) -> TimeStep[Observation]:
        extras = timestep.extras
        extras["final_observation"] = timestep.observation
        return timestep.replace(extras=extras)

    def _maybe_reset(self, state: State, timestep: TimeStep) -> Tuple[State, TimeStep[Observation]]:
        state, timestep = jax.lax.cond(
            timestep.last(),
            self._auto_reset,
            lambda st, ts: (st, self._obs_in_extras(ts)),
            state,
            timestep,
        )

        return state, timestep


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def setup(self) -> None:
        self.net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(256), nn.relu, nn.Dense(1)])

    @nn.compact
    def __call__(self, x, a):
        x = jnp.concatenate([x, a], axis=-1)
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    n_actions: int

    def setup(self) -> None:
        # n_actions = self.env.action_spec().shape[1]  # (agents, n_actions)
        # n_obs = self.env.observation_spec().agents_view.shape[1]  # (agents, n_obs)

        self.torso = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(256), nn.relu])
        self.mean_nn = nn.Dense(self.n_actions)
        self.logstd_nn = nn.Dense(self.n_actions)

    @nn.compact
    def __call__(self, x: ArrayLike) -> Tuple[Array, Array]:
        x = self.torso(x)

        mean = self.mean_nn(x)

        log_std = self.logstd_nn(x)
        log_std = jnp.tanh(log_std)
        # From SpinUp / Denis Yarats
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std


# todo: inside actor?
@jax.jit
def sample_action(
    mean: ArrayLike, log_std: ArrayLike, key: PRNGKey, action_scale, action_bias, eval: bool = False
) -> Tuple[Array, Array]:
    std = jnp.exp(log_std)
    normal = distrax.Normal(mean, std)

    unbound_action = jax.lax.cond(
        eval,
        lambda: mean,
        lambda: normal.sample(seed=key),
    )
    bound_action = jnp.tanh(unbound_action)
    scaled_action = bound_action * action_scale + action_bias

    log_prob = normal.log_prob(unbound_action)
    log_prob -= jnp.log(action_scale * (1 - bound_action**2) + 1e-6)
    log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)

    # we don't use this, but leaving here in case we need it
    # mean = jnp.tanh(mean) * self.action_scale + self.action_bias

    return scaled_action, log_prob


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    logger = neptune.init_run(project="InstaDeep/mava", tags=["sac", args.env_id])

    key = jax.random.PRNGKey(args.seed)

    envs = jaxmarl.make(args.env_id, homogenisation_method="max", auto_reset=False)
    envs = JaxMarlWrapper(envs)
    envs = RecordEpisodeMetrics(envs)
    envs = VmapAutoResetWrapper(envs)

    n_agents = envs.action_spec().shape[0]
    action_dim = envs.action_spec().shape[1]
    obs_dim = envs.observation_spec().agents_view.shape[1]
    full_action_shape = (args.n_envs, n_agents, action_dim)

    act_high = jnp.zeros(envs.action_spec().shape) + envs.action_spec().maximum
    act_low = jnp.zeros(envs.action_spec().shape) + envs.action_spec().minimum
    action_scale = jnp.array((act_high - act_low) / 2.0)[jnp.newaxis, :]
    action_bias = jnp.array((act_high + act_low) / 2.0)[jnp.newaxis, :]

    key, actor_key, q1_key, q2_key, q1_target_key, q2_target_key = jax.random.split(key, 6)

    dummy_actions = jnp.zeros((1, n_agents, action_dim))
    dummy_obs = jnp.zeros((1, n_agents, obs_dim))

    actor = Actor(action_dim)
    actor_params = actor.init(actor_key, dummy_obs)

    q = SoftQNetwork()
    q1_params = q.init(q1_key, dummy_obs, dummy_actions)
    q2_params = q.init(q2_key, dummy_obs, dummy_actions)
    q1_target_params = q.init(q1_target_key, dummy_obs, dummy_actions)
    q2_target_params = q.init(q2_target_key, dummy_obs, dummy_actions)

    actor_opt = optax.adam(args.policy_lr)
    actor_opt_state = actor_opt.init(actor_params)
    q_opt = optax.adam(args.q_lr)
    q_opt_state = q_opt.init((q1_params, q2_params))

    # Automatic entropy tuning
    if args.autotune:
        # -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        target_entropy = jnp.repeat(-n_agents * action_dim, n_agents).astype(float)
        # making sure we have dim (B,A,X) so broacasting works fine
        target_entropy = target_entropy[jnp.newaxis, :, jnp.newaxis]
        log_alpha = jnp.zeros_like(target_entropy)
        alpha = jnp.exp(log_alpha)
        alpha_opt = optax.adam(args.q_lr)
        alpha_opt_state = alpha_opt.init(log_alpha)
    else:
        alpha = args.alpha

    class Transition(NamedTuple):
        obs: ArrayLike
        action: ArrayLike
        reward: ArrayLike
        done: ArrayLike
        next_obs: ArrayLike

    dummy_transition = Transition(
        obs=jnp.zeros((n_agents, obs_dim), dtype=float),
        action=jnp.zeros((n_agents, action_dim), dtype=float),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jnp.zeros((n_agents, obs_dim), dtype=float),
    )

    rb = fbx.make_flat_buffer(
        max_length=args.buffer_size,
        min_length=args.learning_starts,
        sample_batch_size=args.batch_size,
        add_batch_size=args.n_envs,
    )
    buffer_state = rb.init(dummy_transition)
    # buffer_add = jax.jit(rb.add, donate_argnums=0)
    buffer_sample = jax.jit(rb.sample)

    start_time = time.time()
    key = jax.random.PRNGKey(0)

    # jit forward passes
    actor_apply = jax.jit(actor.apply)
    q_apply = jax.jit(q.apply)

    @chex.assert_max_traces(n=2)
    def step(action, obs, env_state, buffer_state):
        env_state, timestep = env_step(env_state, action)
        next_obs = timestep.observation.agents_view
        rewards = timestep.reward[:, 0]
        # todo logical not
        terms = (1 - timestep.discount).astype(bool)[:, 0]
        truncs = timestep.last()
        infos = timestep.extras

        real_next_obs = infos["final_observation"].agents_view

        transition = Transition(obs, action, rewards, terms, real_next_obs)
        buffer_state = rb.add(buffer_state, transition)

        return next_obs, env_state, buffer_state, infos["episode_metrics"]

    @jax.jit
    @chex.assert_max_traces(n=1)
    def explore(_, carry):
        (obs, env_state, buffer_state, metrics, key) = carry
        key, explore_key = jax.random.split(key)
        action = jax.random.uniform(explore_key, full_action_shape)
        next_obs, env_state, buffer_state, metrics = step(action, obs, env_state, buffer_state)

        return next_obs, env_state, buffer_state, metrics, key

    @jax.jit
    @chex.assert_max_traces(n=1)
    def act(actor_params, obs, key, env_state, buffer_state):
        mean, log_std = actor_apply(actor_params, obs)
        action, _ = sample_action(mean, log_std, key, action_scale, action_bias)

        next_obs, env_state, buffer_state, infos = step(action, obs, env_state, buffer_state)
        return next_obs, env_state, buffer_state, infos

    # losses:
    @jax.jit
    @chex.assert_max_traces(n=1)
    def q_loss_fn(q_params, obs, action, target):
        q1_params, q2_params = q_params
        q1_a_values = q.apply(q1_params, obs, action).reshape(-1)
        q2_a_values = q.apply(q2_params, obs, action).reshape(-1)

        q1_loss = jnp.mean((q1_a_values - target) ** 2)
        q2_loss = jnp.mean((q2_a_values - target) ** 2)

        loss = q1_loss + q2_loss

        return loss, (loss, q1_loss, q2_loss, q1_a_values, q2_a_values)

    @jax.jit
    @chex.assert_max_traces(n=1)
    def actor_loss_fn(actor_params, obs, alpha, q_params, key):
        q1_params, q2_params = q_params

        mean, log_std = actor.apply(actor_params, obs)
        pi, log_pi = sample_action(mean, log_std, key, action_scale, action_bias)

        qf1_pi = q.apply(q1_params, obs, pi)
        qf2_pi = q.apply(q2_params, obs, pi)
        min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)

        return ((alpha * log_pi) - min_qf_pi).mean()

    @jax.jit
    @chex.assert_max_traces(n=1)
    def alpha_loss_fn(log_alpha, log_pi, target_entropy):
        return jnp.mean(-jnp.exp(log_alpha) * (log_pi + target_entropy))

    @partial(
        jax.jit,
        donate_argnames=["q_params", "q_target_params", "actor_params", "log_alpha", "opt_states"],
    )
    @chex.assert_max_traces(n=1)
    def train(
        q_params, q_target_params, actor_params, log_alpha, opt_states, data, key, target_entropy
    ):
        q1_params, q2_params = q_params
        q1_target_params, q2_target_params = q_target_params
        q_opt_state, actor_opt_state, alpha_opt_state = opt_states
        target_key, actor_key, alpha_key = jax.random.split(key, 3)

        mean, log_std = actor_apply(actor_params, data.next_obs)
        next_state_actions, next_state_log_pi = sample_action(
            mean, log_std, target_key, action_scale, action_bias
        )

        qf1_next_target = q_apply(q1_target_params, data.next_obs, next_state_actions)
        qf2_next_target = q_apply(q2_target_params, data.next_obs, next_state_actions)
        min_qf_next_target = (
            jnp.minimum(qf1_next_target, qf2_next_target) - jnp.exp(log_alpha) * next_state_log_pi
        )

        rewards = data.reward[..., jnp.newaxis, jnp.newaxis]
        dones = data.done[..., jnp.newaxis, jnp.newaxis]
        next_q_value = (rewards + (1.0 - dones) * args.gamma * min_qf_next_target).reshape(-1)

        # optimize the q networks
        q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
        q_grads, (q_loss, q1_loss, q2_loss, q1_a_vals, q2_a_vals) = q_grad_fn(
            (q1_params, q2_params), data.obs, data.action, next_q_value
        )
        q_updates, q_opt_state = q_opt.update(q_grads, q_opt_state)
        q1_params, q2_params = optax.apply_updates((q1_params, q2_params), q_updates)

        # if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
        # compensate for the delay by doing 'actor_update_interval' instead of 1
        # for _ in range(args.policy_frequency):
        actor_grad_fn = jax.value_and_grad(actor_loss_fn)
        actor_loss, act_grads = actor_grad_fn(
            actor_params,
            data.obs,
            jnp.exp(log_alpha),
            (q1_params, q2_params),
            actor_key,
        )
        actor_updates, actor_opt_state = actor_opt.update(act_grads, actor_opt_state)
        actor_params = optax.apply_updates(actor_params, actor_updates)

        if args.autotune:
            mean, log_std = actor_apply(actor_params, data.obs)
            _, log_pi = sample_action(mean, log_std, alpha_key, action_scale, action_bias)

            alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
            alpha_loss, alpha_grads = alpha_grad_fn(log_alpha, log_pi, target_entropy)
            alpha_updates, alpha_opt_state = alpha_opt.update(alpha_grads, alpha_opt_state)
            log_alpha = optax.apply_updates(log_alpha, alpha_updates)

        q1_target_params = optax.incremental_update(q1_params, q1_target_params, args.tau)
        q2_target_params = optax.incremental_update(q2_params, q2_target_params, args.tau)

        return (
            (q1_params, q2_params),
            (q1_target_params, q2_target_params),
            actor_params,
            log_alpha,
            (q_opt_state, actor_opt_state, alpha_opt_state),
            (q_loss, q1_loss, q2_loss, q1_a_vals, q2_a_vals, actor_loss, alpha_loss),
        )

    @jax.jit
    @chex.assert_max_traces(n=1)
    def _act_and_learn(carry, _):
        (
            obs,
            env_state,
            buffer_state,
            q_params,
            q_target_params,
            actor_params,
            log_alpha,
            opt_states,
            key,
        ) = carry

        key, act_key, buff_key, learn_key = jax.random.split(key, 4)
        next_obs, env_state, buffer_state, metrics = act(
            actor_params, obs, act_key, env_state, buffer_state
        )

        data = buffer_sample(buffer_state, buff_key).experience.first

        (
            q_params,
            q_target_params,
            actor_params,
            log_alpha,
            opt_states,
            losses,
        ) = train(
            q_params,
            q_target_params,
            actor_params,
            log_alpha,
            opt_states,
            data,
            learn_key,
            target_entropy,
        )
        # losses = (q_loss, q1_loss, q2_loss, q1_a_vals, q2_a_vals, actor_loss, alpha_loss),

        return (
            (
                next_obs,
                env_state,
                buffer_state,
                q_params,
                q_target_params,
                actor_params,
                log_alpha,
                opt_states,
                key,
            ),
            (metrics, losses),
        )

    # TRY NOT TO MODIFY: start the game
    keys = jax.random.split(key, args.n_envs)
    env_reset = jax.jit(envs.reset)
    env_step = jax.jit(envs.step)

    @jax.jit
    @chex.assert_max_traces(n=1)
    def first_explore(first_timestep, env_state, buffer_state, key):
        init_val = (
            first_timestep.observation.agents_view,
            env_state,
            buffer_state,
            first_timestep.extras["episode_metrics"],
            key,
        )
        return jax.lax.fori_loop(0, args.learning_starts // args.n_envs, explore, init_val)

    start_time = time.time()
    state, timestep = env_reset(keys)
    # obs = timestep.observation.agents_view

    ep_return = 0.0

    # fill up buffer
    next_obs, env_state, buffer_state, metrics, key = first_explore(
        timestep, state, buffer_state, key
    )
    print("first explore done")
    mean_return = np.mean(metrics["episode_return"])
    print(
        f"[{args.learning_starts}] return: {mean_return:.3f} | sps: {args.learning_starts / (time.time() - start_time):.3f}"
    )

    learner_state = (
        next_obs,
        env_state,
        buffer_state,
        (q1_params, q2_params),
        (q1_target_params, q2_target_params),
        actor_params,
        log_alpha,
        (q_opt_state, actor_opt_state, alpha_opt_state),
        key,
    )

    steps_between_logging = 128
    for global_step in range(args.learning_starts, args.total_timesteps):
        t = global_step * args.n_envs
        learner_state, (metrics, losses) = jax.lax.scan(
            _act_and_learn, learner_state, None, length=steps_between_logging, unroll=16
        )

        mean_return = np.mean(metrics["episode_return"])
        sps = t / (time.time() - start_time)

        logger["mean episode return"].log(mean_return, step=t)
        logger["steps per second"].log(sps, step=t)
        print(f"[{t}] return: {mean_return:.3f} | sps: {t:.3f}")

    # for global_step in range(0, args.total_timesteps, args.n_envs):
    #     # ALGO LOGIC: put action logic here
    #     key, act_key = jax.random.split(key)
    #     if global_step < args.learning_starts:
    #         actions = jax.random.uniform(act_key, (args.n_envs, *envs.action_spec().shape))
    #         # actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
    #     else:
    #         mean, log_std = actor_apply(actor_params, obs)
    #         actions, _ = sample_action(mean, log_std, act_key, action_scale, action_bias)
    #
    #     # TRY NOT TO MODIFY: execute the game and log data.
    #     state, timestep = env_step(state, actions)
    #     next_obs = timestep.observation.agents_view
    #     rewards = timestep.reward
    #     terminations = 1 - timestep.discount
    #     truncations = timestep.last()
    #     infos = timestep.extras
    #
    #     ep_return += np.mean(rewards)
    #     # next_obs, rewards, terminations, truncations, infos = envs.step(actions)
    #
    #     # TRY NOT TO MODIFY: record rewards for plotting purposes
    #     # if "final_info" in infos:
    #     #     for info in infos["final_info"]:
    #     #         if info is None or "episode" not in info:
    #     #             continue
    #     #         print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
    #     #         tensorboard_logger.log_value(
    #     #             "charts/episodic_return", info["episode"]["r"], global_step
    #     #         )
    #     #         tensorboard_logger.log_value(
    #     #             "charts/episodic_length", info["episode"]["l"], global_step
    #     #         )
    #     #         break
    #     if timestep.last()[0]:
    #         logger["episode return"].log(ep_return, step=global_step)
    #         print(f"{global_step}: {ep_return}")
    #         ep_return = 0.0
    #
    #     # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
    #     # real_next_obs = next_obs.copy()
    #     # for idx, trunc in enumerate(truncations):
    #     #     if trunc:
    #     #         real_next_obs[idx] = infos["final_observation"][idx]
    #     real_next_obs = infos["final_observation"].agents_view
    #
    #     transition = Transition(
    #         obs, actions, rewards[:, 0], terminations.astype(bool)[:, 0], real_next_obs
    #     )
    #     buffer_state = buffer_add(buffer_state, transition)
    #
    #     # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
    #     obs = next_obs
    #
    #     # ALGO LOGIC: training.
    #     if global_step > args.learning_starts:
    #         key, buff_key, target_key, actor_key, alpha_key = jax.random.split(key, 5)
    #         _data = buffer_sample(buffer_state, buff_key)
    #         data = _data.experience.first
    #
    #         (
    #             (q1_params, q2_params),
    #             (q1_target_params, q2_target_params),
    #             actor_params,
    #             log_alpha,
    #             (q_opt_state, actor_opt_state, alpha_opt_state),
    #             (q_loss, q1_loss, q2_loss, q1_a_vals, q2_a_vals, actor_loss, alpha_loss),
    #         ) = train(
    #             (q1_params, q2_params),
    #             (q1_target_params, q2_target_params),
    #             actor_params,
    #             log_alpha,
    #             (q_opt_state, actor_opt_state, alpha_opt_state),
    #             data,
    #             (target_key, actor_key, alpha_key),
    #             target_entropy,
    #         )
    #
    #         alpha = jnp.exp(log_alpha)
    #
    #         # update the target networks
    #         # if global_step % args.target_network_frequency == 0:
    #         # q1_target_params = optax.incremental_update(q1_params, q1_target_params, args.tau)
    #         # q2_target_params = optax.incremental_update(q2_params, q2_target_params, args.tau)
    #
    #         if global_step % 100 == 0:
    #             logger["losses/qf1_values"].log(np.mean(q1_a_vals).item(), step=global_step)
    #             logger["losses/qf2_values"].log(np.mean(q2_a_vals).item(), step=global_step)
    #             logger["losses/qf1_loss"].log(q1_loss.item(), step=global_step)
    #             logger["losses/qf2_loss"].log(q2_loss.item(), step=global_step)
    #             logger["losses/qf_loss"].log((q_loss / 2.0).item(), step=global_step)
    #             logger["losses/actor_loss"].log(actor_loss.item(), step=global_step)
    #             logger["losses/mean_alpha"].log(np.mean(alpha).item(), step=global_step)
    #             # print("SPS:", int(global_step / (time.time() - start_time)))
    #             logger["charts/SPS"].log(
    #                 int(global_step / (time.time() - start_time)), step=global_step
    #             )
    #             if args.autotune:
    #                 logger["losses/alpha_loss"].log(alpha_loss.item(), step=global_step)

    # envs.close()
    # writer.close()
