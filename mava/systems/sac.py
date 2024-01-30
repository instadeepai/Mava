import os
import time
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple, Tuple

import chex
import distrax
import flashbax as fbx
import flax
import flax.linen as nn
import jax

# jax.config.update("jax_platform_name", "cpu")
import jax.numpy as jnp
import jaxmarl
import neptune
import numpy as np
import optax
import tyro
from chex import PRNGKey
from flashbax.buffers.flat_buffer import TrajectoryBuffer
from flax.core.scope import FrozenVariableDict
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
    n_envs: int = 256
    """number of parallel environments"""


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
    key: PRNGKey


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
    def __call__(self, x, a):
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

    logger = neptune.init_run(project="InstaDeep/mava", tags=["sac", "pmapped", args.env_id])

    key = jax.random.PRNGKey(args.seed)
    devices = jax.devices()
    n_devices = len(devices)

    env = jaxmarl.make(args.env_id, homogenisation_method="max", auto_reset=False)
    env = JaxMarlWrapper(env)
    env = RecordEpisodeMetrics(env)
    env = VmapAutoResetWrapper(env)

    n_agents = env.action_spec().shape[0]
    action_dim = env.action_spec().shape[1]
    obs_dim = env.observation_spec().agents_view.shape[1]
    full_action_shape = (args.n_envs, n_agents, action_dim)

    act_high = jnp.zeros(env.action_spec().shape) + env.action_spec().maximum
    act_low = jnp.zeros(env.action_spec().shape) + env.action_spec().minimum
    action_scale = jnp.array((act_high - act_low) / 2.0)[jnp.newaxis, :]
    action_bias = jnp.array((act_high + act_low) / 2.0)[jnp.newaxis, :]

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
    target_entropy = jnp.repeat(-action_dim, n_agents).astype(float)
    # making sure we have dim=3 so broacasting works fine
    target_entropy = target_entropy[jnp.newaxis, :, jnp.newaxis]
    log_alpha = jnp.zeros_like(target_entropy)
    alpha = jnp.exp(log_alpha)

    # Pack params
    online_q_params = Qs(q1_params, q2_params)
    target_q_params = Qs(q1_target_params, q2_target_params)
    params = SacParams(actor_params, QsAndTarget(online_q_params, target_q_params), log_alpha)

    # Make opt states.
    actor_opt = optax.adam(args.policy_lr)
    actor_opt_state = actor_opt.init(params.actor)

    q_opt = optax.adam(args.q_lr)
    q_opt_state = q_opt.init(params.q.online)

    alpha_opt = optax.adam(args.q_lr)  # todo: alpha lr?
    alpha_opt_state = alpha_opt.init(params.log_alpha)

    # Pack opt states
    opt_states = OptStates(actor_opt_state, q_opt_state, alpha_opt_state)

    # Distribute params and opt states across all devices
    params = jax.device_put_replicated(params, devices)
    opt_states = jax.device_put_replicated(opt_states, devices)

    # todo: move
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
    buffer_state = jax.device_put_replicated(rb.init(dummy_transition), devices)
    buffer_sample = jax.jit(rb.sample)

    start_time = time.time()
    key = jax.random.PRNGKey(0)

    # jit forward passes
    actor_apply = jax.jit(actor.apply)
    q_apply = jax.jit(q.apply)

    @chex.assert_max_traces(n=2)
    def step(action, obs, env_state, buffer_state):
        env_state, timestep = env.step(env_state, action)
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

    @chex.assert_max_traces(n=1)
    def explore(_, carry):
        (obs, env_state, buffer_state, metrics, key) = carry
        key, explore_key = jax.random.split(key)
        action = jax.random.uniform(explore_key, full_action_shape)
        next_obs, env_state, buffer_state, metrics = step(action, obs, env_state, buffer_state)

        return next_obs, env_state, buffer_state, metrics, key

    @chex.assert_max_traces(n=1)
    def act(actor_params, obs, key, env_state, buffer_state):
        mean, log_std = actor_apply(actor_params, obs)
        action, _ = sample_action(mean, log_std, key, action_scale, action_bias)

        next_obs, env_state, buffer_state, infos = step(action, obs, env_state, buffer_state)
        return next_obs, env_state, buffer_state, infos

    # losses:
    @jax.jit
    @chex.assert_max_traces(n=1)
    def q_loss_fn(q_params: Qs, obs, action, target):
        q1_params, q2_params = q_params
        q1_a_values = q.apply(q1_params, obs, action).reshape(-1)
        q2_a_values = q.apply(q2_params, obs, action).reshape(-1)

        q1_loss = jnp.mean((q1_a_values - target) ** 2)
        q2_loss = jnp.mean((q2_a_values - target) ** 2)

        loss = q1_loss + q2_loss

        return loss, (loss, q1_loss, q2_loss, q1_a_values, q2_a_values)

    @jax.jit
    @chex.assert_max_traces(n=1)
    def actor_loss_fn(actor_params, obs, alpha, q_params: Qs, key):
        mean, log_std = actor.apply(actor_params, obs)
        pi, log_pi = sample_action(mean, log_std, key, action_scale, action_bias)

        qf1_pi = q.apply(q_params.q1, obs, pi)
        qf2_pi = q.apply(q_params.q2, obs, pi)
        min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)

        return ((alpha * log_pi) - min_qf_pi).mean()

    @jax.jit
    @chex.assert_max_traces(n=1)
    def alpha_loss_fn(log_alpha, log_pi, target_entropy):
        return jnp.mean(-jnp.exp(log_alpha) * (log_pi + target_entropy))

    @partial(
        jax.jit,
        donate_argnames=["params", "opt_states"],
    )
    @chex.assert_max_traces(n=1)
    def train(
        params: SacParams, opt_states: OptStates, data, key, target_entropy
    ) -> Tuple[SacParams, optax.OptState, tuple]:  # todo: typing
        target_key, actor_key, alpha_key = jax.random.split(key, 3)

        # Generate Q target values.
        mean, log_std = actor_apply(params.actor, data.next_obs)
        next_state_actions, next_state_log_pi = sample_action(
            mean, log_std, target_key, action_scale, action_bias
        )

        qf1_next_target = q_apply(params.q.targets.q1, data.next_obs, next_state_actions)
        qf2_next_target = q_apply(params.q.targets.q2, data.next_obs, next_state_actions)
        min_qf_next_target = (
            jnp.minimum(qf1_next_target, qf2_next_target)
            - jnp.exp(params.log_alpha) * next_state_log_pi
        )

        rewards = data.reward[..., jnp.newaxis, jnp.newaxis]
        dones = data.done[..., jnp.newaxis, jnp.newaxis]
        next_q_value = (rewards + (1.0 - dones) * args.gamma * min_qf_next_target).reshape(-1)

        # Update Q.
        q_grad_fn = jax.grad(q_loss_fn, has_aux=True)
        q_grads, q_loss_info = q_grad_fn(params.q.online, data.obs, data.action, next_q_value)
        q_grads, q_loss_info = jax.lax.pmean((q_grads, q_loss_info), axis_name="device")
        q_updates, new_q_opt_state = q_opt.update(q_grads, opt_states.q)
        new_online_q_params = optax.apply_updates(params.q.online, q_updates)

        # Update actor.
        # if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
        # compensate for the delay by doing 'actor_update_interval' instead of 1
        # for _ in range(args.policy_frequency):
        actor_grad_fn = jax.value_and_grad(actor_loss_fn)
        actor_loss, act_grads = actor_grad_fn(
            params.actor,
            data.obs,
            jnp.exp(params.log_alpha),
            new_online_q_params,
            actor_key,
        )
        actor_loss, act_grads = jax.lax.pmean((actor_loss, act_grads), axis_name="device")
        actor_updates, new_actor_opt_state = actor_opt.update(act_grads, opt_states.actor)
        new_actor_params = optax.apply_updates(params.actor, actor_updates)

        # Update alpha.
        # if args.autotune:
        mean, log_std = actor_apply(new_actor_params, data.obs)
        _, log_pi = sample_action(mean, log_std, alpha_key, action_scale, action_bias)

        alpha_grad_fn = jax.value_and_grad(alpha_loss_fn)
        alpha_loss, alpha_grads = alpha_grad_fn(params.log_alpha, log_pi, target_entropy)
        alpha_loss, alpha_grads = jax.lax.pmean((alpha_loss, alpha_grads), axis_name="device")
        alpha_updates, new_alpha_opt_state = alpha_opt.update(alpha_grads, opt_states.alpha)
        new_log_alpha = optax.apply_updates(params.log_alpha, alpha_updates)

        # Target network polyak update.
        new_q1_target_params = optax.incremental_update(
            new_online_q_params.q1, params.q.targets.q1, args.tau
        )
        new_q2_target_params = optax.incremental_update(
            new_online_q_params.q2, params.q.targets.q2, args.tau
        )

        params = SacParams(
            new_actor_params,
            QsAndTarget(new_online_q_params, Qs(new_q1_target_params, new_q2_target_params)),
            new_log_alpha,
        )
        opt_states = OptStates(new_actor_opt_state, new_q_opt_state, new_alpha_opt_state)
        # todo: dict
        q_loss, q1_loss, q2_loss, q1_a_vals, q2_a_vals = q_loss_info
        losses = q_loss, q1_loss, q2_loss, q1_a_vals, q2_a_vals, actor_loss, alpha_loss

        return params, opt_states, losses

    @jax.jit
    @chex.assert_max_traces(n=1)  # todo: typing
    def _act_and_learn(carry: LearnerState, _) -> Tuple[LearnerState, tuple]:
        """Act, sample, learn."""
        obs, env_state, buffer_state, params, opt_states, key = carry
        key, act_key, buff_key, learn_key = jax.random.split(key, 4)

        # Act
        next_obs, env_state, buffer_state, metrics = act(
            params.actor, obs, act_key, env_state, buffer_state
        )

        # Sample
        data = buffer_sample(buffer_state, buff_key).experience.first

        # Learn
        params, opt_states, losses = train(params, opt_states, data, learn_key, target_entropy)

        return (
            LearnerState(next_obs, env_state, buffer_state, params, opt_states, key),
            (metrics, losses),
        )

    # TRY NOT TO MODIFY: start the game
    reset_keys = jax.random.split(key, args.n_envs * n_devices)
    reset_keys = jnp.reshape(reset_keys, (n_devices, args.n_envs, -1))

    start_time = time.time()
    env_state, first_timestep = jax.pmap(env.reset, axis_name="device")(reset_keys)

    ep_return = 0.0

    # fill up buffer
    explore_keys = jax.random.split(key, n_devices)
    init_explore_state = (
        first_timestep.observation.agents_view,
        env_state,
        buffer_state,
        first_timestep.extras["episode_metrics"],
        explore_keys,
    )
    pmaped_explore = jax.pmap(
        lambda state: jax.lax.fori_loop(0, args.learning_starts // args.n_envs, explore, state),
        axis_name="device",
    )
    next_obs, env_state, buffer_state, metrics, key = pmaped_explore(init_explore_state)

    print("first explore done")
    ep_returns = metrics["episode_return"][metrics["episode_return"] != 0]
    mean_return = np.mean(ep_returns)
    print(
        f"[{args.learning_starts}] return: {mean_return:.3f} | sps: {args.learning_starts / (time.time() - start_time):.3f}"
    )

    # rollout lenght?
    steps_between_logging = 10_000
    learner_state = LearnerState(next_obs, env_state, buffer_state, params, opt_states, key)
    pmapped_learn = jax.pmap(
        lambda state: jax.lax.scan(_act_and_learn, state, None, length=steps_between_logging),
        axis_name="device",
    )

    for t in range(0, args.total_timesteps, args.n_envs):
        learner_state, (metrics, losses) = pmapped_learn(learner_state)

        ep_returns = metrics["episode_return"][metrics["episode_return"] != 0]
        mean_return = np.mean(ep_returns)
        max_return = np.max(ep_returns)

        q_loss, q1_loss, q2_loss, q1_a_vals, q2_a_vals, actor_loss, alpha_loss = losses
        log_alpha = learner_state.params.log_alpha

        curr_step = t * steps_between_logging + args.learning_starts
        sps = curr_step / (time.time() - start_time)

        logger["mean episode return"].log(mean_return, step=curr_step)
        logger["max episode return"].log(max_return, step=curr_step)

        logger["q loss"].log(np.mean(q_loss), step=curr_step)
        logger["q1 loss"].log(np.mean(q1_loss), step=curr_step)
        logger["q2 loss"].log(np.mean(q2_loss), step=curr_step)
        logger["q1 values"].log(np.mean(q1_a_vals), step=curr_step)
        logger["q2 values"].log(np.mean(q2_a_vals), step=curr_step)
        logger["actor loss"].log(np.mean(actor_loss), step=curr_step)
        logger["alpha loss"].log(np.mean(alpha_loss), step=curr_step)
        logger["alpha"].log(np.mean(np.exp(learner_state.params.log_alpha)), step=curr_step)

        logger["steps per second"].log(sps, step=curr_step)

        print(f"[{curr_step}] return: {mean_return:.3f} | sps: {sps:.3f}")
