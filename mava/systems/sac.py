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
from jumanji.types import Observation
from jumanji.wrappers import GymObservation, jumanji_to_gym_obs

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
    env_id: str = "HalfCheetah"
    """the environment id of the task"""
    factorization: str = "2x3"
    """how the joints are split up"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    learning_starts: int = 5e3
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


def batchify(obs):
    return np.stack(list(obs.values()))


def unbatchify(obs):
    return {f"agent_{i}": o for i, o in enumerate(obs)}


class MaMuJoCoWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

        self.num_agents = len(self.env.observation_spaces)

        obs_space = self.env.observation_space("agent_0")
        new_shape = (self.num_agents, obs_space.shape[0])
        self.observation_space = Box(
            low=np.full(new_shape, obs_space.low[0]),
            high=np.full(new_shape, obs_space.high[0]),
            shape=new_shape,
        )

        act_space = self.env.action_space("agent_0")
        new_shape = (self.num_agents, act_space.shape[0])
        self.action_space = Box(
            low=np.full(new_shape, act_space.low[0]),
            high=np.full(new_shape, act_space.high[0]),
            shape=new_shape,
        )

    def reset(self, seed, options):
        obs, info = self.env.reset()
        obs = batchify(obs)

        return obs, info

    def step(self, action):
        o, r, term, trunc, i = self.env.step(unbatchify(action))
        term = batchify(term)
        trunc = batchify(trunc)

        return batchify(o), batchify(r)[0], np.all(term), np.all(trunc), i

    def state(self):
        return self.env.state()


def make_env(env_id, factorization):
    def thunk():
        # env = gymnasium_robotics.mamujoco_v0.parallel_env(env_id, factorization)
        # env = MaMuJoCoWrapper(env)
        env = jaxmarl.make("halfcheetah-6x1")
        env = JaxMarlWrapper(env)
        env = JumanjiToGymWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


class JumanjiToGymWrapper(gym.Env):
    """A wrapper that converts a Jumanji `Environment` to one that follows the `gym.Env` API."""

    # Flag that prevents `gym.register` from misinterpreting the `_step` and
    # `_reset` as signs of a deprecated gym Env API.
    _gym_disable_underscore_compat: ClassVar[bool] = True

    def __init__(self, env: Environment, seed: int = 0, backend: Optional[str] = None):
        """Create the Gym environment.

        Args:
            env: `Environment` to wrap to a `gym.Env`.
            seed: the seed that is used to initialize the environment's PRNG.
            backend: the XLA backend.
        """
        self._env = env
        self.metadata: Dict[str, str] = {}
        self._key = jax.random.PRNGKey(seed)
        self.backend = backend
        self._state = None
        self.observation_space = specs.jumanji_specs_to_gym_spaces(self._env.observation_spec())
        self.action_space = specs.jumanji_specs_to_gym_spaces(self._env.action_spec())

        def reset(key: chex.PRNGKey) -> Tuple[State, Observation, Optional[Dict]]:
            """Reset function of a Jumanji environment to be jitted."""
            state, timestep = self._env.reset(key)
            return state, timestep.observation, timestep.extras

        self._reset = jax.jit(reset, backend=self.backend)

        def step(
            state: State, action: chex.Array
        ) -> Tuple[State, Observation, chex.Array, bool, Optional[Any]]:
            """Step function of a Jumanji environment to be jitted."""
            state, timestep = self._env.step(state, action)
            trunc = jnp.bool_(timestep.last())
            term = bool(1 - timestep.discount)
            return state, timestep.observation, timestep.reward, term, trunc, timestep.extras

        self._step = jax.jit(step, backend=self.backend)

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        return_info: bool = False,
        options: Optional[dict] = None,
    ) -> Union[GymObservation, Tuple[GymObservation, Optional[Any]]]:
        """Resets the environment to an initial state by starting a new sequence
        and returns the first `Observation` of this sequence.

        Returns:
            obs: an element of the environment's observation_space.
            info (optional): contains supplementary information such as metrics.
        """
        if seed is not None:
            self.seed(seed)
        key, self._key = jax.random.split(self._key)
        self._state, obs, extras = self._reset(key)

        # Convert the observation to a numpy array or a nested dict thereof
        obs = jumanji_to_gym_obs(obs)

        if return_info:
            info = jax.tree_util.tree_map(np.asarray, extras)
            return obs, info
        else:
            return obs, {}  # type: ignore

    def step(
        self, action: chex.ArrayNumpy
    ) -> Tuple[GymObservation, float, bool, bool, Optional[Any]]:
        """Updates the environment according to the action and returns an `Observation`.

        Args:
            action: A NumPy array representing the action provided by the agent.

        Returns:
            observation: an element of the environment's observation_space.
            reward: the amount of reward returned as a result of taking the action.
            terminated: whether a terminal state is reached.
            info: contains supplementary information such as metrics.
        """

        action = jnp.array(action)  # Convert input numpy array to JAX array
        self._state, obs, reward, term, trunc, extras = self._step(self._state, action)

        # Convert to get the correct signature
        obs = jumanji_to_gym_obs(obs)
        reward = float(reward)
        term = bool(term)
        trunc = bool(trunc)
        info = jax.tree_util.tree_map(np.asarray, extras)

        return obs, reward, term, trunc, info

    def seed(self, seed: int = 0) -> None:
        """Function which sets the seed for the environment's random number generator(s).

        Args:
            seed: the seed value for the random number generator(s).
        """
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode: str = "human") -> Any:
        """Renders the environment.

        Args:
            mode: currently not used since Jumanji does not currently support modes.
        """
        del mode
        return self._env.render(self._state)

    def close(self) -> None:
        """Closes the environment, important for rendering where pygame is imported."""
        self._env.close()

    @property
    def unwrapped(self) -> Environment:
        return self._env


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
    env: gym.vector.VectorEnv

    def setup(self) -> None:
        n_actions = self.env.single_action_space.shape[1]  # (agents, n_actions)
        n_obs = self.env.single_observation_space.shape[1]  # (agents, n_obs)

        self.torso = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(256), nn.relu])
        self.mean_nn = nn.Dense(n_actions)
        self.logstd_nn = nn.Dense(n_actions)

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
    import stable_baselines3 as sb3

    if sb3.__version__ < "2.0":
        raise ValueError(
            """Ongoing migration: run the following command to install the new dependencies:
poetry run pip install "stable_baselines3==2.0.0a1"
"""
        )

    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    tensorboard_logger.configure(run_name)
    # writer.add_text(
    #     "hyperparameters",
    #     "|param|value|\n|-|-|\n%s"
    #     % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    # )

    # TRY NOT TO MODIFY: seeding
    key = jax.random.PRNGKey(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, args.factorization) for _ in range(args.n_envs)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    envs.single_observation_space.dtype = np.float32

    n_agents = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[1]
    obs_dim = envs.single_observation_space.shape[1]

    # state_dim = envs.call("state")[0].shape[0]
    act_high = envs.single_action_space.high
    act_low = envs.single_action_space.low
    action_scale = jnp.array((act_high - act_low) / 2.0)[jnp.newaxis, :]
    action_bias = jnp.array((act_high + act_low) / 2.0)[jnp.newaxis, :]

    key, actor_key, q1_key, q2_key, q1_target_key, q2_target_key = jax.random.split(key, 6)

    dummy_actions = jnp.zeros((1, n_agents, action_dim))
    dummy_obs = jnp.zeros((1, n_agents, obs_dim))

    actor = Actor(envs)
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
    buffer_add = jax.jit(rb.add, donate_argnums=0)
    buffer_sample = jax.jit(rb.sample)

    start_time = time.time()
    key = jax.random.PRNGKey(0)

    # jit forward passes
    actor_apply = jax.jit(actor.apply)
    q_apply = jax.jit(q.apply)

    # losses:
    @jax.jit
    def q_loss_fn(q_params, obs, action, target):
        q1_params, q2_params = q_params
        q1_a_values = q.apply(q1_params, obs, action).reshape(-1)
        q2_a_values = q.apply(q2_params, obs, action).reshape(-1)

        q1_loss = jnp.mean((q1_a_values - target) ** 2)
        q2_loss = jnp.mean((q2_a_values - target) ** 2)

        loss = q1_loss + q2_loss

        return loss, (loss, q1_loss, q2_loss, q1_a_values, q2_a_values)

    @jax.jit
    def actor_loss_fn(actor_params, obs, alpha, q_params, key):
        q1_params, q2_params = q_params

        mean, log_std = actor.apply(actor_params, obs)
        pi, log_pi = sample_action(mean, log_std, key, action_scale, action_bias)

        qf1_pi = q.apply(q1_params, obs, pi)
        qf2_pi = q.apply(q2_params, obs, pi)
        min_qf_pi = jnp.minimum(qf1_pi, qf2_pi)

        return ((alpha * log_pi) - min_qf_pi).mean()

    @jax.jit
    def alpha_loss_fn(log_alpha, log_pi, target_entropy):
        return jnp.mean(-jnp.exp(log_alpha) * (log_pi + target_entropy))

    @partial(
        jax.jit,
        donate_argnames=["q_params", "q_target_params", "actor_params", "log_alpha", "opt_states"],
    )
    def train(
        q_params, q_target_params, actor_params, log_alpha, opt_states, data, keys, target_entropy
    ):
        q1_params, q2_params = q_params
        q1_target_params, q2_target_params = q_target_params
        q_opt_state, actor_opt_state, alpha_opt_state = opt_states
        target_key, actor_key, alpha_key = keys

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

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    for global_step in range(0, args.total_timesteps, args.n_envs):
        # ALGO LOGIC: put action logic here
        key, act_key = jax.random.split(key)
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            mean, log_std = actor_apply(actor_params, obs)
            actions, _ = sample_action(mean, log_std, act_key, action_scale, action_bias)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is None or "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                tensorboard_logger.log_value(
                    "charts/episodic_return", info["episode"]["r"], global_step
                )
                tensorboard_logger.log_value(
                    "charts/episodic_length", info["episode"]["l"], global_step
                )
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        transition = Transition(obs, actions, rewards, terminations, real_next_obs)
        buffer_state = buffer_add(buffer_state, transition)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            key, buff_key, target_key, actor_key, alpha_key = jax.random.split(key, 5)
            _data = buffer_sample(buffer_state, buff_key)
            data = _data.experience.first

            (
                (q1_params, q2_params),
                (q1_target_params, q2_target_params),
                actor_params,
                log_alpha,
                (q_opt_state, actor_opt_state, alpha_opt_state),
                (q_loss, q1_loss, q2_loss, q1_a_vals, q2_a_vals, actor_loss, alpha_loss),
            ) = train(
                (q1_params, q2_params),
                (q1_target_params, q2_target_params),
                actor_params,
                log_alpha,
                (q_opt_state, actor_opt_state, alpha_opt_state),
                data,
                (target_key, actor_key, alpha_key),
                target_entropy,
            )

            alpha = jnp.exp(log_alpha)

            # update the target networks
            # if global_step % args.target_network_frequency == 0:
            # q1_target_params = optax.incremental_update(q1_params, q1_target_params, args.tau)
            # q2_target_params = optax.incremental_update(q2_params, q2_target_params, args.tau)

            if global_step % 100 == 0:
                tensorboard_logger.log_value(
                    "losses/qf1_values", jnp.mean(q1_a_vals).item(), global_step
                )
                tensorboard_logger.log_value(
                    "losses/qf2_values", jnp.mean(q2_a_vals).item(), global_step
                )
                tensorboard_logger.log_value("losses/qf1_loss", q1_loss.item(), global_step)
                tensorboard_logger.log_value("losses/qf2_loss", q2_loss.item(), global_step)
                tensorboard_logger.log_value("losses/qf_loss", (q_loss / 2.0).item(), global_step)
                tensorboard_logger.log_value("losses/actor_loss", actor_loss.item(), global_step)
                tensorboard_logger.log_value(
                    "losses/mean_alpha", jnp.mean(alpha).item(), global_step
                )
                print("SPS:", int(global_step / (time.time() - start_time)))
                tensorboard_logger.log_value(
                    "charts/SPS", int(global_step / (time.time() - start_time)), global_step
                )
                if args.autotune:
                    tensorboard_logger.log_value(
                        "losses/alpha_loss", alpha_loss.item(), global_step
                    )

    envs.close()
    # writer.close()
