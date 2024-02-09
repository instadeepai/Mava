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

import time
from datetime import datetime
from typing import Any, Callable, Dict, NamedTuple, Tuple

import chex
import distrax
import flashbax as fbx
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import jaxmarl
import optax
from brax.io import html
from chex import PRNGKey
from colorama import Fore, Style
from flashbax.buffers.flat_buffer import TrajectoryBuffer
from flashbax.buffers.trajectory_buffer import TrajectoryBufferState
from flax.core.scope import FrozenVariableDict
from flax.jax_utils import unreplicate
from jax import Array
from jax.typing import ArrayLike
from jumanji.env import State
from jumanji.types import Observation, TimeStep
from jumanji.wrappers import Wrapper
from omegaconf import DictConfig, OmegaConf
from typing_extensions import TypeAlias

from mava.utils.logger import LogEvent, MavaLogger
from mava.wrappers import AgentIDWrapper, JaxMarlWrapper, episode_metrics
from mava.wrappers.jumanji import MultiAgentWrapper

# jax.config.update("jax_platform_name", "cpu")


# todo: types.py
class Qs(NamedTuple):
    q1: FrozenVariableDict
    q2: FrozenVariableDict


class QsAndTarget(NamedTuple):
    online: Qs
    targets: Qs


class SacParams(NamedTuple):
    actor: FrozenVariableDict
    q: QsAndTarget
    log_alpha: Array


class OptStates(NamedTuple):
    actor: optax.OptState
    q: optax.OptState
    alpha: optax.OptState


class LearnerState(NamedTuple):
    obs: Array
    env_state: State
    buffer_state: TrajectoryBuffer
    params: SacParams
    opt_states: OptStates
    t: int
    key: PRNGKey


class Transition(NamedTuple):
    obs: ArrayLike
    action: ArrayLike
    reward: ArrayLike
    done: ArrayLike
    next_obs: ArrayLike


Metrics = Dict[str, Array]

BufferState: TypeAlias = TrajectoryBufferState[Transition]


class AutoResetWrapper(Wrapper):
    """Automatically resets environments that are done. Once the terminal state is reached,
    the state, observation, and step_type are reset. The observation and step_type of the
    terminal TimeStep is reset to the reset observation and StepType.LAST, respectively.
    The reward, discount, and extras retrieved from the transition to the terminal state.
    WARNING: do not `jax.vmap` the wrapped environment (e.g. do not use with the `VmapWrapper`),
    which would lead to inefficient computation due to both the `step` and `reset` functions
    being processed each time `step` is called. Please use the `VmapAutoResetWrapper` instead.
    """

    # todo: make this take in state also
    def _obs_in_extras(self, timestep: TimeStep[Observation]) -> TimeStep[Observation]:
        extras = timestep.extras
        extras["final_observation"] = timestep.observation
        return timestep.replace(extras=extras)

    def _auto_reset(self, state: State, timestep: TimeStep) -> Tuple[State, TimeStep[Observation]]:
        """Reset the state and overwrite `timestep.observation` with the reset observation
        if the episode has terminated.
        """
        if not hasattr(state, "key"):
            raise AttributeError(
                "This wrapper assumes that the state has attribute key which is used"
                " as the source of randomness for automatic reset"
            )

        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        key, _ = jax.random.split(state.key)  # type: ignore
        state, reset_timestep = self._env.reset(key)

        extras = timestep.extras
        extras["final_observation"] = timestep.observation

        # Replace observation with reset observation.
        timestep = timestep.replace(  # type: ignore
            observation=reset_timestep.observation, extras=extras
        )

        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Step the environment, with automatic resetting if the episode terminates."""
        state, timestep = self._env.step(state, action)

        # Overwrite the state and timestep appropriately if the episode terminates.
        state, timestep = jax.lax.cond(
            timestep.last(),
            self._auto_reset,
            lambda st, ts: (st, self._obs_in_extras(ts)),
            state,
            timestep,
        )

        return state, timestep


# todo: jumanji PR
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
    def __call__(self, x: Array, a: Array) -> Array:
        x = jnp.concatenate([x, a], axis=-1)
        return self.net(x)


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    n_actions: int

    def setup(self) -> None:
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
def sample_action(
    mean: ArrayLike,
    log_std: ArrayLike,
    key: PRNGKey,
    action_scale: Array,
    action_bias: Array,
    eval: bool = False,
) -> Tuple[Array, Array]:
    std = jnp.exp(log_std)
    # normal = distrax.Normal(mean, std)
    normal = distrax.MultivariateNormalDiag(mean, std)

    unbound_action = jax.lax.cond(
        eval,
        lambda: mean,
        lambda: normal.sample(seed=key),
    )
    bound_action = jnp.tanh(unbound_action)
    scaled_action = bound_action * action_scale + action_bias

    # MultivariateNormalDiag sums on last dim
    # log_prob = normal.log_prob(unbound_action)
    # rescale_term = jnp.log(action_scale * (1 - bound_action**2) + 1e-6).sum(axis=-1)
    # log_prob = log_prob - rescale_term

    # from: https://github.com/google-deepmind/distrax/issues/216
    # todo: scale
    log_prob = normal.log_prob(unbound_action) - jnp.sum(
        2 * (jnp.log(2) - unbound_action - jax.nn.softplus(-2 * unbound_action)), axis=-1
    )

    # we don't use this, but leaving here in case we need it
    # mean = jnp.tanh(mean) * self.action_scale + self.action_bias

    return scaled_action, log_prob[..., jnp.newaxis]


# env, nns, opts, rb, params, opt_states, buffer_state, target_entropy, logger, key
def init(
    cfg: DictConfig,
) -> Tuple[
    MultiAgentWrapper,
    Tuple[Actor, SoftQNetwork],
    Tuple[optax.GradientTransformation, optax.GradientTransformation, optax.GradientTransformation],
    TrajectoryBuffer,
    SacParams,
    OptStates,
    BufferState,
    chex.Array,
    MavaLogger,
    chex.PRNGKey,
]:
    logger = MavaLogger(cfg)

    key = jax.random.PRNGKey(cfg.system.seed)
    devices = jax.devices()

    env = jaxmarl.make(cfg.env.env_name, **cfg.env.kwargs)
    env = JaxMarlWrapper(env)
    env = AgentIDWrapper(env)
    env = AutoResetWrapper(env)
    env = episode_metrics.RecordEpisodeMetrics(env)

    n_agents = env.action_spec().shape[0]
    action_dim = env.action_spec().shape[1]
    obs_dim = env.observation_spec().agents_view.shape[1]

    key, actor_key, q1_key, q2_key, q1_target_key, q2_target_key = jax.random.split(key, 6)

    dummy_actions = jnp.zeros((1, n_agents, action_dim))
    dummy_obs = jnp.zeros((1, n_agents, obs_dim))

    # Making actor network
    actor = Actor(action_dim)
    actor_params = actor.init(actor_key, dummy_obs)

    # Making Q networks
    q = SoftQNetwork()
    q1_params = q.init(q1_key, dummy_obs, dummy_actions)
    q2_params = q.init(q2_key, dummy_obs, dummy_actions)
    q1_target_params = q.init(q1_target_key, dummy_obs, dummy_actions)
    q2_target_params = q.init(q2_target_key, dummy_obs, dummy_actions)

    # Automatic entropy tuning
    target_entropy = -cfg.system.target_entropy_scale * action_dim
    target_entropy = jnp.repeat(target_entropy, n_agents).astype(float)
    # making sure we have dim=3 so broacasting works fine
    target_entropy = target_entropy[jnp.newaxis, :, jnp.newaxis]
    if cfg.system.autotune:
        log_alpha = jnp.zeros_like(target_entropy)
    else:
        log_alpha = jnp.log(cfg.system.init_alpha)
        log_alpha = jnp.broadcast_to(log_alpha, target_entropy.shape)

    # Pack params
    online_q_params = Qs(q1_params, q2_params)
    target_q_params = Qs(q1_target_params, q2_target_params)
    params = SacParams(actor_params, QsAndTarget(online_q_params, target_q_params), log_alpha)

    # Make opt states.
    actor_opt = optax.adam(cfg.system.policy_lr)
    actor_opt_state = actor_opt.init(params.actor)

    q_opt = optax.adam(cfg.system.q_lr)
    q_opt_state = q_opt.init(params.q.online)

    alpha_opt = optax.adam(cfg.system.q_lr)  # todo: alpha lr?
    alpha_opt_state = alpha_opt.init(params.log_alpha)

    # Pack opt states
    opt_states = OptStates(actor_opt_state, q_opt_state, alpha_opt_state)

    # Distribute params and opt states across all devices
    params = jax.device_put_replicated(params, devices)
    opt_states = jax.device_put_replicated(opt_states, devices)

    dummy_transition = Transition(
        obs=jnp.zeros((n_agents, obs_dim), dtype=float),
        action=jnp.zeros((n_agents, action_dim), dtype=float),
        reward=jnp.zeros((), dtype=float),
        done=jnp.zeros((), dtype=bool),
        next_obs=jnp.zeros((n_agents, obs_dim), dtype=float),
    )

    rb = fbx.make_flat_buffer(
        max_length=cfg.system.buffer_size,
        min_length=cfg.system.explore_steps,
        sample_batch_size=cfg.system.batch_size,
        add_batch_size=cfg.system.n_envs,
    )
    buffer_state = jax.device_put_replicated(rb.init(dummy_transition), devices)

    nns = (actor, q)
    opts = (actor_opt, q_opt, alpha_opt)

    return env, nns, opts, rb, params, opt_states, buffer_state, target_entropy, logger, key


def make_learn(
    cfg: DictConfig,
    env: MultiAgentWrapper,
    nns: Tuple[Actor, SoftQNetwork],
    opts: Tuple[
        optax.GradientTransformation, optax.GradientTransformation, optax.GradientTransformation
    ],
    rb: TrajectoryBuffer,
    target_entropy: chex.Array,
) -> Tuple[
    Callable[
        [int, Tuple[Array, State, BufferState, Dict, chex.PRNGKey]],
        Tuple[Array, State, BufferState, Dict, chex.PRNGKey],
    ],
    Callable[[LearnerState, Any], Tuple[LearnerState, Tuple[Dict[str, Array], Dict[str, Array]]]],
]:
    actor, q = nns
    actor_opt, q_opt, alpha_opt = opts

    full_action_shape = (cfg.system.n_envs, *env.action_spec().shape)

    act_high = jnp.zeros(env.action_spec().shape) + env.action_spec().maximum
    act_low = jnp.zeros(env.action_spec().shape) + env.action_spec().minimum
    action_scale = jnp.array((act_high - act_low) / 2.0)[jnp.newaxis, :]
    action_bias = jnp.array((act_high + act_low) / 2.0)[jnp.newaxis, :]

    def step(
        action: Array, obs: Array, env_state: State, buffer_state: BufferState
    ) -> Tuple[Array, State, BufferState, Dict]:
        env_state, timestep = jax.vmap(env.step)(env_state, action)
        next_obs = timestep.observation.agents_view
        rewards = timestep.reward[:, 0]
        # todo logical not
        terms = (1 - timestep.discount).astype(bool)[:, 0]
        # truncs = timestep.last()
        infos = timestep.extras

        real_next_obs = infos["final_observation"].agents_view

        transition = Transition(obs, action, rewards, terms, real_next_obs)
        buffer_state = rb.add(buffer_state, transition)

        return next_obs, env_state, buffer_state, infos["episode_metrics"]

    # losses:
    def q_loss_fn(q_params: Qs, obs: Array, action: Array, target: Array) -> Tuple[Array, Metrics]:
        q1_params, q2_params = q_params
        q1_a_values = q.apply(q1_params, obs, action).reshape(-1)
        q2_a_values = q.apply(q2_params, obs, action).reshape(-1)

        q1_loss = jnp.mean((q1_a_values - target) ** 2)
        q2_loss = jnp.mean((q2_a_values - target) ** 2)

        loss = q1_loss + q2_loss
        loss_info = {
            "loss": loss,
            "q1_loss": q1_loss,
            "q2_loss": q2_loss,
            "q1_a_vals": q1_a_values,
            "q2_a_vals": q2_a_values,
        }

        return loss, loss_info

    def actor_loss_fn(
        actor_params: FrozenVariableDict, obs: Array, alpha: Array, q_params: Qs, key: chex.PRNGKey
    ) -> Array:
        mean, log_std = actor.apply(actor_params, obs)
        pi, log_pi = sample_action(mean, log_std, key, action_scale, action_bias)

        qf1_pi = q.apply(q_params.q1, obs, pi)
        qf2_pi = q.apply(q_params.q2, obs, pi)
        min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)

        return ((alpha * log_pi) - min_qf_pi).mean()

    def alpha_loss_fn(log_alpha: Array, log_pi: Array, target_entropy: Array) -> Array:
        return jnp.mean(-jnp.exp(log_alpha) * (log_pi + target_entropy))

    def update_q(
        params: SacParams, opt_states: OptStates, data: Transition, key: chex.PRNGKey
    ) -> Tuple[SacParams, OptStates, Metrics]:
        # Generate Q target values.
        mean, log_std = actor.apply(params.actor, data.next_obs)
        next_state_actions, next_state_log_pi = sample_action(
            mean, log_std, key, action_scale, action_bias
        )

        qf1_next_target = q.apply(params.q.targets.q1, data.next_obs, next_state_actions)
        qf2_next_target = q.apply(params.q.targets.q2, data.next_obs, next_state_actions)
        min_qf_next_target = (
            jnp.minimum(qf1_next_target, qf2_next_target)
            - jnp.exp(params.log_alpha) * next_state_log_pi
        )

        rewards = data.reward[..., jnp.newaxis, jnp.newaxis]
        dones = data.done[..., jnp.newaxis, jnp.newaxis]
        next_q_value = (rewards + (1.0 - dones) * cfg.system.gamma * min_qf_next_target).reshape(-1)

        # Update Q.
        q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(params.q.online, data.obs, data.action, next_q_value)
        q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")
        q_updates, new_q_opt_state = q_opt.update(q_grads, opt_states.q)
        new_online_q_params = optax.apply_updates(params.q.online, q_updates)

        # Target network polyak update.
        new_target_q_params = optax.incremental_update(
            new_online_q_params, params.q.targets, cfg.system.tau
        )

        q_and_target = QsAndTarget(new_online_q_params, new_target_q_params)
        params = params._replace(q=q_and_target)
        opt_states = opt_states._replace(q=new_q_opt_state)

        return params, opt_states, q_loss_info

    def update_actor_and_alpha(
        params: SacParams, opt_states: OptStates, data: Transition, key: chex.PRNGKey
    ) -> Tuple[SacParams, OptStates, Metrics]:
        # compensate for the delay by doing `policy_frequency` updates instead of 1.
        assert cfg.system.policy_frequency > 0
        for _ in range(cfg.system.policy_frequency):
            actor_key, alpha_key = jax.random.split(key)

            # Update actor.
            actor_grad_fn = jax.value_and_grad(actor_loss_fn)
            actor_loss, act_grads = actor_grad_fn(
                params.actor, data.obs, jnp.exp(params.log_alpha), params.q.online, actor_key
            )
            actor_loss, act_grads = jax.lax.pmean((actor_loss, act_grads), axis_name="device")
            actor_updates, new_actor_opt_state = actor_opt.update(act_grads, opt_states.actor)
            new_actor_params = optax.apply_updates(params.actor, actor_updates)

            # Update alpha if autotuning.
            if cfg.system.autotune:
                mean, log_std = actor.apply(new_actor_params, data.obs)
                _, log_pi = sample_action(mean, log_std, alpha_key, action_scale, action_bias)

                alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
                alpha_loss, alpha_grads = alpha_grad_fn(params.log_alpha, log_pi, target_entropy)
                alpha_loss, alpha_grads = jax.lax.pmean(
                    (alpha_loss, alpha_grads), axis_name="device"
                )
                alpha_updates, new_alpha_opt_state = alpha_opt.update(alpha_grads, opt_states.alpha)
                new_log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)

                params = params._replace(actor=new_actor_params, log_alpha=new_log_alpha)
                opt_states = opt_states._replace(
                    actor=new_actor_opt_state, alpha=new_alpha_opt_state
                )
            else:
                params = params._replace(actor=new_actor_params)
                opt_states = opt_states._replace(actor=new_actor_opt_state)
                alpha_loss = 0.0

        loss_info = {"actor_loss": actor_loss, "alpha_loss": alpha_loss}
        return params, opt_states, loss_info

    def update(
        params: SacParams, opt_states: OptStates, data: Transition, t: int, key: chex.PRNGKey
    ) -> Tuple[SacParams, OptStates, Metrics]:
        q_key, actor_key = jax.random.split(key)

        params, opt_states, q_loss_info = update_q(params, opt_states, data, q_key)
        params, opt_states, act_loss_info = jax.lax.cond(
            t % cfg.system.policy_frequency == 0,  # TD 3 Delayed update support
            update_actor_and_alpha,
            # just return same params and opt_states and 0 for losses
            lambda params, opt_states, *_: (
                params,
                opt_states,
                {"actor_loss": 0.0, "alpha_loss": 0.0},
            ),
            params,
            opt_states,
            data,
            actor_key,
        )

        losses = q_loss_info | act_loss_info
        return params, opt_states, losses

    def sample_and_learn(
        carry: Tuple[BufferState, SacParams, OptStates, int, chex.PRNGKey], _: Any
    ) -> Tuple[Tuple[BufferState, SacParams, OptStates, int, chex.PRNGKey], Metrics]:
        buffer_state, params, opt_states, t, key = carry
        key, buff_key, learn_key = jax.random.split(key, 3)

        data = rb.sample(buffer_state, buff_key).experience.first
        params, opt_state, losses = update(params, opt_states, data, t, learn_key)

        return (buffer_state, params, opt_state, t, key), losses

    scanned_learn = lambda state: jax.lax.scan(
        sample_and_learn, state, None, length=cfg.system.learn_steps
    )

    # todo: make this scannable, not for_i
    def explore(
        _: int, carry: Tuple[Array, State, BufferState, Dict, chex.PRNGKey]
    ) -> Tuple[Array, State, BufferState, Dict, chex.PRNGKey]:
        (obs, env_state, buffer_state, metrics, key) = carry
        key, explore_key = jax.random.split(key)
        action = jax.random.uniform(explore_key, full_action_shape)
        next_obs, env_state, buffer_state, metrics = step(action, obs, env_state, buffer_state)

        return next_obs, env_state, buffer_state, metrics, key

    def act(
        carry: Tuple[FrozenVariableDict, Array, State, BufferState, chex.PRNGKey], _: Any
    ) -> Tuple[Tuple[FrozenVariableDict, Array, State, BufferState, chex.PRNGKey], Dict]:
        actor_params, obs, env_state, buffer_state, key = carry

        mean, log_std = actor.apply(actor_params, obs)
        action, _ = sample_action(mean, log_std, key, action_scale, action_bias)

        next_obs, env_state, buffer_state, metrics = step(action, obs, env_state, buffer_state)
        return (actor_params, next_obs, env_state, buffer_state, key), metrics

    scanned_act = lambda state: jax.lax.scan(act, state, None, length=cfg.system.act_steps)

    # todo: typing
    def act_and_learn(
        carry: LearnerState, _: Any
    ) -> Tuple[LearnerState, Tuple[Dict[str, Array], Dict[str, Array]]]:
        """Act, sample, learn."""

        obs, env_state, buffer_state, params, opt_states, t, key = carry
        key, act_key, learn_key = jax.random.split(key, 3)
        # Act
        act_state = (params.actor, obs, env_state, buffer_state, act_key)
        (_, next_obs, env_state, buffer_state, _), metrics = scanned_act(act_state)

        # Sample and learn
        learn_state = (buffer_state, params, opt_states, t, learn_key)
        (buffer_state, params, opt_states, _, _), losses = scanned_learn(learn_state)

        return (
            LearnerState(next_obs, env_state, buffer_state, params, opt_states, t + 1, key),
            (metrics, losses),
        )

    return explore, act_and_learn


def run_experiment(cfg: DictConfig) -> Array:
    n_devices = len(jax.devices())

    pmaped_steps = 1024  # todo: config option
    steps_btwn_log = n_devices * cfg.system.n_envs * cfg.system.act_steps * pmaped_steps

    env, nns, opts, rb, params, opt_states, buffer_state, target_entropy, logger, key = init(cfg)
    explore, act_and_learn = make_learn(cfg, env, nns, opts, rb, target_entropy)

    # todo: do this in make_learn?
    pmaped_learn = jax.pmap(
        lambda state: jax.lax.scan(act_and_learn, state, None, length=pmaped_steps),
        axis_name="device",
        donate_argnums=0,
    )

    pmaped_explore = jax.pmap(
        lambda state: jax.lax.fori_loop(
            0, cfg.system.explore_steps // cfg.system.n_envs, explore, state
        ),
        axis_name="device",
        donate_argnums=0,
    )

    start_time = time.time()

    reset_keys = jax.random.split(key, cfg.system.n_envs * n_devices)
    reset_keys = jnp.reshape(reset_keys, (n_devices, cfg.system.n_envs, -1))

    env_state, first_timestep = jax.pmap(jax.vmap(env.reset), axis_name="device")(reset_keys)

    # fill up buffer/explore
    explore_keys = jax.random.split(key, n_devices)
    init_explore_state = (
        first_timestep.observation.agents_view,
        env_state,
        buffer_state,
        first_timestep.extras["episode_metrics"],
        explore_keys,
    )
    next_obs, env_state, buffer_state, metrics, key = pmaped_explore(init_explore_state)

    # log explore episode metrics
    final_metrics = episode_metrics.get_final_step_metrics(metrics)
    logger.log(final_metrics, cfg.system.explore_steps, 0, LogEvent.ACT)

    # Initial learner state.
    learner_state = LearnerState(
        next_obs, env_state, buffer_state, params, opt_states, jnp.zeros(n_devices), key
    )

    # Main loop:
    # We want start to align with the final step of the first pmaped_learn,
    # where we've done explore_steps and 1 full learn step.
    start = cfg.system.explore_steps + steps_btwn_log
    for t in range(start, int(cfg.system.total_timesteps), steps_btwn_log):
        # learn
        learner_state, (metrics, losses) = pmaped_learn(learner_state)
        # eval:
        # todo...

        # Multiply by learn steps here because anakin steps per second is learn + act steps
        # But we want to make sure we're counting env steps correctly so it's not included
        # in the loop counter.
        sps = t * cfg.system.learn_steps / (time.time() - start_time)
        final_metrics = episode_metrics.get_final_step_metrics(metrics)

        logger.log({"step": t, "steps per second": sps}, t, 0, LogEvent.MISC)
        logger.log(final_metrics, t, 0, LogEvent.ACT)
        logger.log(losses | {"log_alpha": learner_state.params.log_alpha}, t, 0, LogEvent.TRAIN)

    # Record video
    states = []
    actor_params = unreplicate(learner_state.params.actor)

    # todo: store this in the actor?
    act_high = jnp.zeros(env.action_spec().shape) + env.action_spec().maximum
    act_low = jnp.zeros(env.action_spec().shape) + env.action_spec().minimum
    action_scale = jnp.array((act_high - act_low) / 2.0)[jnp.newaxis, :]
    action_bias = jnp.array((act_high + act_low) / 2.0)[jnp.newaxis, :]

    actor_apply = jax.jit(nns[0].apply)
    env_step = jax.jit(env.step)
    sample_action_jit = jax.jit(sample_action)

    _key, reset_key = jax.random.split(jax.random.PRNGKey(1))
    state, ts = env.reset(reset_key)

    for _ in range(1000):
        _key, step_key = jax.random.split(_key)

        mean, log_std = actor_apply(actor_params, ts.observation.agents_view)
        action, _ = sample_action_jit(mean, log_std, step_key, action_scale, action_bias)
        state, ts = env_step(state, action.squeeze(0))
        states.append(state)

    anim_file = f"anim_{cfg.env.env_name}_{datetime.now().strftime('%Y%m%d%H%M%S')}.html"
    html.save(anim_file, env.unwrapped.sys, [s.env_state.state.pipeline_state for s in states])

    return jnp.mean(metrics["episode_return"])


@hydra.main(config_path="../configs", config_name="default_ff_isac.yaml", version_base="1.2")
def hydra_entry_point(cfg: DictConfig) -> float:
    """Experiment entry point."""
    # Allow dynamic attributes.
    OmegaConf.set_struct(cfg, False)

    # Run experiment.
    final_return = run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}ISAC experiment completed{Style.RESET_ALL}")

    return float(final_return)


if __name__ == "__main__":
    hydra_entry_point()
