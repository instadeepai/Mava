# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/sac/#sac_continuous_actionpy
import os
import random
import time
from dataclasses import dataclass
from typing import NamedTuple, Tuple

import distrax
import flashbax as fbx
import flax.linen as nn
import gymnasium as gym
import gymnasium_robotics
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tyro
from chex import PRNGKey
from gymnasium.spaces import Box, MultiDiscrete
from jax import Array
from jax.typing import ArrayLike
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
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
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
    alpha: float = 0.2
    """Entropy regularization coefficient."""
    autotune: bool = True
    """automatic tuning of the entropy coefficient"""
    n_envs: int = 8
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

    # @property
    # def observation_space(self) -> gym.Space:
    #     space = self.env.observation_space("agent_0")
    #     new_shape = (self.num_agents, space.shape[0])
    #     return Box(
    #         low=np.full(new_shape, space.low[0]),
    #         high=np.full(new_shape, space.high[0]),
    #         shape=new_shape,
    #     )

    # @property
    # def action_space(self) -> gym.spaces.Space:
    #     space = self.env.action_space("agent_0")
    #     new_shape = (self.num_agents, space.shape[0])
    #     return Box(
    #         low=np.full(new_shape, space.low[0]),
    #         high=np.full(new_shape, space.high[0]),
    #         shape=new_shape,
    #     )


def make_env(env_id, factorization):
    # def thunk():
    #     if capture_video and idx == 0:
    #         env = gym.make(env_id, render_mode="rgb_array")
    #         env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
    #     else:
    #         env = gym.make(env_id)
    #     env = gym.wrappers.RecordEpisodeStatistics(env)
    #     env.action_space.seed(seed)
    #     return env
    #
    # return thunk
    def thunk():
        env = gymnasium_robotics.mamujoco_v0.parallel_env(env_id, factorization)
        env = MaMuJoCoWrapper(env)
        env = gym.wrappers.RecordEpisodeStatistics(env)

        return env

    return thunk


# ALGO LOGIC: initialize agent here:
class SoftQNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        # todo: slightly better prod(shape[1:])?
        # n_actions = env.single_action_space.shape[1]  # (agents, n_actions)
        # n_obs = env.single_observation_space.shape[1]  # (agents, n_obs)
        #
        # self.fc1 = nn.Linear(n_obs + n_actions, 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc3 = nn.Linear(256, 1)

        self.net = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(256), nn.relu, nn.Dense(1)])

    def __call__(self, x, a):
        x = jnp.concatenate([x, a], axis=-1)
        return self.net(x)

    # def forward(self, x, a):
    #     x = torch.cat([x, a], -1)
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     return x


LOG_STD_MAX = 2
LOG_STD_MIN = -5


class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        # todo: slightly better prod(shape[1:])?
        n_actions = env.single_action_space.shape[1]  # (agents, n_actions)
        n_obs = env.single_observation_space.shape[1]  # (agents, n_obs)

        # self.fc1 = nn.Linear(n_obs, 256)
        # self.fc2 = nn.Linear(256, 256)
        # self.fc_mean = nn.Linear(256, n_actions)
        # self.fc_logstd = nn.Linear(256, n_actions)  # action rescaling

        self.torso = nn.Sequential([nn.Dense(256), nn.relu, nn.Dense(256), nn.relu])
        self.mean_nn = nn.Dense(n_actions)
        self.logstd_nn = nn.Dense(n_actions)

        # self.register_buffer(
        #     "action_scale",
        #     torch.tensor(
        #         (env.single_action_space.high - env.single_action_space.low) / 2.0,
        #         dtype=torch.float32,
        #     ).unsqueeze(0),
        # )
        # self.register_buffer(
        #     "action_bias",
        #     torch.tensor(
        #         (env.single_action_space.high + env.single_action_space.low) / 2.0,
        #         dtype=torch.float32,
        #     ).unsqueeze(0),
        # )
        act_high = env.single_action_space.high
        act_low = env.single_action_space.low
        self.action_scale = jnp.array((act_high - act_low) / 2.0)[jnp.newaxis, :]
        self.action_bias = jnp.array((act_high + act_low) / 2.0)[jnp.newaxis, :]

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     mean = self.fc_mean(x)
    #     log_std = self.fc_logstd(x)
    #     log_std = torch.tanh(log_std)
    #     # From SpinUp / Denis Yarats
    #     log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)
    #
    #     return mean, log_std

    def __call__(self, x: ArrayLike) -> Tuple[Array, Array]:
        x = self.torso(x)

        mean = self.mean_nn(x)

        log_std = self.logstd_nn(x)
        log_std = jnp.tanh(log_std)
        # From SpinUp / Denis Yarats
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (log_std + 1)

        return mean, log_std

    # def get_action(self, x: ArrayLike) -> Tuple[Array, Array, Array]:
    #     mean, log_std = self(x)
    #     std = log_std.exp()
    #     normal = torch.distributions.Normal(mean, std)
    #     x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
    #     y_t = torch.tanh(x_t)
    #     action = y_t * self.action_scale + self.action_bias
    #     log_prob = normal.log_prob(x_t)
    #     # Enforcing Action Bound
    #     log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
    #     log_prob = log_prob.sum(axis=-1, keepdim=True)
    #     mean = torch.tanh(mean) * self.action_scale + self.action_bias
    #     return action, log_prob, mean

    def sample_action(self, x: ArrayLike, key: PRNGKey, eval: bool = False) -> Tuple[Array, Array]:
        mean, log_std = self(x)
        std = jnp.exp(log_std)
        normal = distrax.Normal(mean, std)

        unbound_action = jax.lax.cond(
            eval,
            lambda: mean,
            lambda: normal.sample(seed=key),
        )
        bound_action = jnp.tanh(unbound_action)
        scaled_action = bound_action * self.action_scale + self.action_bias

        log_prob = normal.log_prob(unbound_action)
        log_prob -= jnp.log(self.action_scale * (1 - bound_action**2) + 1e-6)
        log_prob = jnp.sum(log_prob, axis=-1, keepdims=True)

        # we don't use this, but leaving here in case we need it
        # mean = jnp.tanh(mean) * self.action_scale + self.action_bias

        return scaled_action, log_prob


# def jax_to_torch(x: Array):
#     return torch.from_numpy(np.array(x)).to("cuda")


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
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    key = jax.random.PRNGKey(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.backends.cudnn.deterministic = args.torch_deterministic

    # device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.AsyncVectorEnv(
        [make_env(args.env_id, args.factorization) for _ in range(args.n_envs)]
    )
    assert isinstance(
        envs.single_action_space, gym.spaces.Box
    ), "only continuous action space is supported"

    # max_action = float(
    #     envs.single_action_space.high[0, 0]
    # )  # hack for now, space is (agents, act_shape)

    # actor = Actor(envs).to(device)
    # qf1 = SoftQNetwork(envs).to(device)
    # qf2 = SoftQNetwork(envs).to(device)
    # qf1_target = SoftQNetwork(envs).to(device)
    # qf2_target = SoftQNetwork(envs).to(device)
    # qf1_target.load_state_dict(qf1.state_dict())
    # qf2_target.load_state_dict(qf2.state_dict())
    # q_optimizer = optim.Adam(list(qf1.parameters()) + list(qf2.parameters()), lr=args.q_lr)
    # actor_optimizer = optim.Adam(list(actor.parameters()), lr=args.policy_lr)
    key, actor_key, q1_key, q2_key, q1_target_key, q2_target_key = jax.random.split(key, 6)

    n_actions = envs.single_action_space.shape[1]  # (agents, n_actions)
    n_obs = envs.single_observation_space.shape[1]  # (agents, n_obs)
    n_agents = envs.single_observation_space.shape[0]

    actor = Actor(envs)
    actor_params = actor.init(actor_key, jnp.zeros((1, n_agents, n_obs)))

    q = SoftQNetwork()
    dummy_q_input = jnp.zeros((1, n_agents, n_obs + n_actions))
    q1_params = q.init(q1_key, dummy_q_input)
    q2_params = q.init(q2_key, dummy_q_input)
    q1_target_params = q.init(q1_target_key, dummy_q_input)
    q2_target_params = q.init(q2_target_key, dummy_q_input)

    actor_optimizer = optax.adam(args.policy_lr)
    actor_opt_state = actor_optimizer.init(actor_params)
    q_optimizer = optax.adam(args.q_lr)
    q_opt_state = q_optimizer.init((q1_params, q2_params))

    # Automatic entropy tuning
    if args.autotune:
        # -torch.prod(torch.Tensor(envs.single_action_space.shape).to(device)).item()
        target_entropy = jnp.array(envs.action_space.shape[-1])
        log_alpha = jnp.zeros(1)
        alpha = jnp.exp(log_alpha)
        a_optimizer = optax.adam(args.q_lr)  # optim.Adam([log_alpha], lr=args.q_lr)
        a_opt_state = a_optimizer.init(log_alpha)
    else:
        alpha = args.alpha

    envs.single_observation_space.dtype = np.float32

    n_agents = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[1]
    obs_dim = envs.single_observation_space.shape[1]
    # state_dim = envs.call("state")[0].shape[0]

    class Transition(NamedTuple):
        obs: ArrayLike
        action: ArrayLike
        reward: ArrayLike
        done: ArrayLike
        next_obs: ArrayLike
        # state: array
        # next_state: array

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
    # rb = ReplayBuffer(
    #     args.buffer_size,
    #     envs.single_observation_space,
    #     envs.single_action_space,
    #     device,
    #     handle_timeout_termination=False,
    # )
    start_time = time.time()
    key = jax.random.PRNGKey(0)

    # TRY NOT TO MODIFY: start the game
    obs, _ = envs.reset()
    for global_step in range(0, args.total_timesteps, args.n_envs):
        # ALGO LOGIC: put action logic here
        if global_step < args.learning_starts:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            actions, _, _ = actor.sample_action(jnp.asarray(obs))
            # actions = actions.detach().cpu().numpy()

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        if "final_info" in infos:
            for info in infos["final_info"]:
                if info is None or "episode" not in info:
                    continue
                print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)
                break

        # TRY NOT TO MODIFY: save data to reply buffer; handle `final_observation`
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc:
                real_next_obs[idx] = infos["final_observation"][idx]

        transition = Transition(obs, actions, rewards, terminations, real_next_obs)
        # transition = jax.tree_map(jnp.array, transition)
        buffer_state = buffer_add(buffer_state, transition)
        # rb.add(obs, real_next_obs, actions, rewards, terminations, infos)

        # TRY NOT TO MODIFY: CRUCIAL step easy to overlook
        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            key, buff_key = jax.random.split(key)
            _data = buffer_sample(buffer_state, buff_key)  # rb.sample(args.batch_size)
            data = _data.experience.first

            # with torch.no_grad():
            next_state_actions, next_state_log_pi, _ = actor.sample_action(data.next_obs)
            qf1_next_target = qf1_target(data.next_obs, next_state_actions)
            qf2_next_target = qf2_target(data.next_obs, next_state_actions)
            min_qf_next_target = (
                jnp.min(qf1_next_target, qf2_next_target) - alpha * next_state_log_pi
            )

            rewards = data.reward.unsqueeze(-1).unsqueeze(-1)
            dones = data.done.unsqueeze(-1).unsqueeze(-1)
            next_q_value = (
                rewards + (1.0 - dones.float()) * args.gamma * min_qf_next_target
            ).reshape(-1)

            def q_loss(q_params, obs, action, target):
                # qf1_a_values = qf1(data.obs, data.action).view(-1)
                # qf2_a_values = qf2(data.obs, data.action).view(-1)
                # qf1_loss = F.mse_loss(qf1_a_values, next_q_value)
                # qf2_loss = F.mse_loss(qf2_a_values, next_q_value)
                # qf_loss = qf1_loss + qf2_loss
                q1_params, q2_params = q_params
                q1_a_values = q.apply(q1_params, obs, action).reshape(-1)
                q2_a_values = q.apply(q2_params, obs, action).reshape(-1)

                q1_loss = jnp.mean((q1_a_values - target) ** 2)
                q2_loss = jnp.mean((q2_a_values - target) ** 2)

                q_loss = q1_loss + q2_loss

                return q_loss, (q1_loss, q2_loss)

            # optimize the q networks
            # q_optimizer.zero_grad()
            # qf_loss.backward()
            # q_optimizer.step()
            q_grad_fn = jax.value_and_grad(q_loss)
            q_grads, (q1_loss, q2_loss) = q_grad_fn(
                (q1_params, q2_params), data.obs, data.action, next_q_value
            )
            q1_params, q2_params = optax.apply_updates(q_grads, (q1_params, q2_params))

            if global_step % args.policy_frequency == 0:  # TD 3 Delayed update support
                for _ in range(
                    args.policy_frequency
                ):  # compensate for the delay by doing 'actor_update_interval' instead of 1
                    pi, log_pi, _ = actor.sample_action(data.obs)
                    qf1_pi = qf1(data.obs, pi)
                    qf2_pi = qf2(data.obs, pi)
                    min_qf_pi = jnp.min(qf1_pi, qf2_pi)
                    actor_loss = ((alpha * log_pi) - min_qf_pi).mean()

                    actor_optimizer.zero_grad()
                    actor_loss.backward()
                    actor_optimizer.step()

                    if args.autotune:
                        with torch.no_grad():
                            _, log_pi, _ = actor.sample_action(data.obs)
                        alpha_loss = (-log_alpha.exp() * (log_pi + target_entropy)).mean()

                        a_optimizer.zero_grad()
                        alpha_loss.backward()
                        a_optimizer.step()
                        alpha = log_alpha.exp().item()

            # update the target networks
            if global_step % args.target_network_frequency == 0:
                for param, target_param in zip(qf1.parameters(), qf1_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )
                for param, target_param in zip(qf2.parameters(), qf2_target.parameters()):
                    target_param.data.copy_(
                        args.tau * param.data + (1 - args.tau) * target_param.data
                    )

            if global_step % 100 == 0:
                writer.add_scalar("losses/qf1_values", qf1_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf2_values", qf2_a_values.mean().item(), global_step)
                writer.add_scalar("losses/qf1_loss", qf1_loss.item(), global_step)
                writer.add_scalar("losses/qf2_loss", qf2_loss.item(), global_step)
                writer.add_scalar("losses/qf_loss", qf_loss.item() / 2.0, global_step)
                writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
                writer.add_scalar("losses/alpha", alpha, global_step)
                print("SPS:", int(global_step / (time.time() - start_time)))
                writer.add_scalar(
                    "charts/SPS", int(global_step / (time.time() - start_time)), global_step
                )
                if args.autotune:
                    writer.add_scalar("losses/alpha_loss", alpha_loss.item(), global_step)

    envs.close()
    writer.close()
