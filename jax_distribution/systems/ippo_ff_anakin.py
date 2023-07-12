"""
An example following the Anakin podracer example found here:
https://colab.research.google.com/drive/1974D-qP17fd5mLxy6QZv-ic4yxlPJp-G?usp=sharing#scrollTo=myLN2J47oNGq
"""

import datetime
import os
import timeit
from os.path import abspath, dirname
from typing import NamedTuple, Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jumanji
import numpy as np
import optax
from flax import struct
from flax.linen.initializers import constant, orthogonal
from jumanji.environments.routing.multi_cvrp.generator import UniformRandomGenerator
from jumanji.environments.routing.multi_cvrp.types import State
from jumanji.types import TimeStep
from jumanji.wrappers import AutoResetWrapper, Wrapper
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.utils import apply_backspaces_and_linefeeds

from jax_distribution.utils.logger_tools import Logger, config_copy, get_logger


class TimeIt:
    def __init__(self, tag, frames=None):
        self.tag = tag
        self.frames = frames

    def __enter__(self):
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args):
        self.elapsed_secs = timeit.default_timer() - self.start
        msg = self.tag + (": Elapsed time=%.2fs" % self.elapsed_secs)
        if self.frames:
            msg += ", FPS=%.2e" % (self.frames / self.elapsed_secs)
        print(msg)


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def reset(self, key: chex.PRNGKey) -> Tuple[LogEnvState, TimeStep]:
        state, timestep = self._env.reset(key)
        # timestep.extras = {}
        state = LogEnvState(state, 0.0, 0, 0.0, 0)
        return state, timestep

    def step(
        self,
        state: State,
        action: jnp.ndarray,
    ) -> Tuple[State, TimeStep]:

        env_state, timestep = self._env.step(state.env_state, action)

        new_episode_return = state.episode_returns + jnp.mean(timestep.reward)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - timestep.last()),
            episode_lengths=new_episode_length * (1 - timestep.last()),
            returned_episode_returns=state.returned_episode_returns
            * (1 - timestep.last())
            + new_episode_return * timestep.last(),
            returned_episode_lengths=state.returned_episode_lengths
            * (1 - timestep.last())
            + new_episode_length * timestep.last(),
        )
        return state, timestep


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    observation: jnp.ndarray
    info: jnp.ndarray


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"
    env_name: str = "RobotWarehouse-v0"

    @nn.compact
    def __call__(self, observation):
        if "MultiCVRP" in self.env_name:
            x = observation.vehicles.coordinates  # .shape
        elif "RobotWarehouse" in self.env_name:
            x = observation.agents_view

        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)

        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_mean)
        actor_mean = activation(actor_mean)

        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_mean,
            jnp.finfo(jnp.float32).min,
        )

        pi = distrax.Categorical(logits=masked_logits)

        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic = activation(critic)
        critic = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic)
        critic = activation(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return pi, jnp.squeeze(critic, axis=-1)


def get_learner_fn(env, forward_pass, opt_update, config):
    def update_step_fn(params, opt_state, outer_rng, env_state, timestep):

        # COLLECT TRAJECTORIES
        def _env_step(runner_state, unused):
            params, env_state, last_timestep, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = forward_pass(params, last_timestep.observation)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            env_state, next_timestep = env.step(env_state, action)

            done, reward = jax.tree_map(
                lambda x: jnp.repeat(x, config["NUM_AGENTS"]).reshape(-1),
                [next_timestep.last(), next_timestep.reward],
            )

            transition = Transition(
                done,
                action,
                value,
                reward,
                log_prob,
                last_timestep.observation,
                {
                    "returned_episode_returns": jnp.repeat(
                        env_state.returned_episode_returns, config["NUM_AGENTS"]
                    ).reshape(-1),
                    "returned_episode_lengths": jnp.repeat(
                        env_state.returned_episode_lengths, config["NUM_AGENTS"]
                    ).reshape(-1),
                    "episode_done": jnp.repeat(
                        next_timestep.last(), config["NUM_AGENTS"]
                    ).reshape(-1),
                },
            )
            runner_state = (params, env_state, next_timestep, rng)
            return runner_state, transition

        runner_state = (params, env_state, timestep, outer_rng)
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
        )

        # CALCULATE ADVANTAGE
        params, env_state, last_timestep, rng = runner_state
        _, last_val = forward_pass(params, last_timestep.observation)

        def _calculate_gae(traj_batch, last_val):
            def _get_advantages(gae_and_next_value, transition):
                gae, next_value = gae_and_next_value
                done, value, reward = (
                    transition.done,
                    transition.value,
                    transition.reward,
                )
                delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                return (gae, value), gae

            _, advantages = jax.lax.scan(
                _get_advantages,
                (jnp.zeros_like(last_val), last_val),
                traj_batch,
                reverse=True,
                unroll=16,
            )
            return advantages, advantages + traj_batch.value

        advantages, targets = _calculate_gae(traj_batch, last_val)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                traj_batch, advantages, targets = batch_info

                def _loss_fn(params, traj_batch, gae, targets):
                    # RERUN NETWORK
                    pi, value = forward_pass(params, traj_batch.observation)
                    log_prob = pi.log_prob(traj_batch.action)

                    # CALCULATE VALUE LOSS
                    value_pred_clipped = traj_batch.value + (
                        value - traj_batch.value
                    ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                    value_losses = jnp.square(value - targets)
                    value_losses_clipped = jnp.square(value_pred_clipped - targets)
                    value_loss = (
                        0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
                    )

                    # CALCULATE ACTOR LOSS
                    ratio = jnp.exp(log_prob - traj_batch.log_prob)
                    gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                    loss_actor1 = ratio * gae
                    loss_actor2 = (
                        jnp.clip(
                            ratio,
                            1.0 - config["CLIP_EPS"],
                            1.0 + config["CLIP_EPS"],
                        )
                        * gae
                    )
                    loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                    loss_actor = loss_actor.mean()
                    entropy = pi.entropy().mean()

                    total_loss = (
                        loss_actor
                        + config["VF_COEF"] * value_loss
                        - config["ENT_COEF"] * entropy
                    )
                    return total_loss, (value_loss, loss_actor, entropy)

                grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                total_loss, grads = grad_fn(params, traj_batch, advantages, targets)
                # pmean
                total_loss = jax.lax.pmean(total_loss, axis_name="i")
                grads = jax.lax.pmean(grads, axis_name="j")
                grads = jax.lax.pmean(grads, axis_name="i")
                updates, new_opt_state = opt_update(grads, opt_state)
                new_params = optax.apply_updates(params, updates)
                return (new_params, new_opt_state), total_loss

            (params, opt_state), traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["ROLLOUT_LENGTH"]
            # TODO: check batch_size
            # assert (
            #     batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
            # ), "batch size must be equal to number of steps * number of envs"
            permutation = jax.random.permutation(_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(
                lambda x: x.reshape((batch_size,) + x.shape[1:]), batch
            )
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=0), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.reshape(
                    x, [config["NUM_MINIBATCHES"], -1] + list(x.shape[1:])
                ),
                shuffled_batch,
            )
            (params, opt_state), total_loss = jax.lax.scan(
                _update_minbatch, (params, opt_state), minibatches
            )

            update_state = ((params, opt_state), traj_batch, advantages, targets, rng)
            return update_state, total_loss

        update_state = ((params, opt_state), traj_batch, advantages, targets, rng)

        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
        )

        ((params, opt_state), traj_batch, advantages, targets, rng) = update_state

        metric_info = jax.tree_util.tree_map(lambda x: x[:, 0], traj_batch.info)
        return params, opt_state, rng, env_state, last_timestep, metric_info

    def update_fn(params, opt_state, rng, env_state, timestep):
        """Compute a gradient update from a single trajectory."""
        rng, loss_rng = jax.random.split(rng)
        (
            new_params,
            new_opt_state,
            rng,
            new_env_state,
            new_timestep,
            metric_info,
        ) = update_step_fn(
            params,
            opt_state,
            rng,
            env_state,
            timestep,
        )

        return (
            new_params,
            new_opt_state,
            rng,
            new_env_state,
            new_timestep,
        ), metric_info

    def learner_fn(params, opt_state, rngs, env_states, timesteps):
        """Vectorise and repeat the update."""
        batched_update_fn = jax.vmap(
            update_fn, axis_name="j"
        )  # vectorize across batch.

        def iterate_fn(val, _):  # repeat many times to avoid going back to Python.
            params, opt_state, rngs, env_states, timesteps = val
            return batched_update_fn(params, opt_state, rngs, env_states, timesteps)

        runner_state = (params, opt_state, rngs, env_states, timesteps)
        runner_state, metric = jax.lax.scan(
            iterate_fn, runner_state, None, config["ITERATIONS"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return learner_fn


def log(logger, metrics_info):
    """Log the episode returns and lengths."""
    dones = jnp.ravel(metrics_info["episode_done"])

    returned_episode_returns = jnp.ravel(metrics_info["returned_episode_returns"])
    returned_episode_lengths = jnp.ravel(metrics_info["returned_episode_lengths"])
    episode_returns = returned_episode_returns[dones]
    episode_lengths = returned_episode_lengths[dones]

    print("MEAN EPISODE RETURN: ", np.mean(episode_returns))

    for ep_i in range(len(episode_returns)):
        logger.log_stat("episode_returns", episode_returns[ep_i], ep_i)
        logger.log_stat("episode_lengths", episode_lengths[ep_i], ep_i)
    logger.log_stat(
        "mean_episode_returns", np.mean(episode_returns), len(episode_returns)
    )


# Logger setup
logger = get_logger()
ex = Experiment("mava", save_git_info=False)
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds
results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.config
def make_config():
    LR = 5e-3
    ENV_NAME = "RobotWarehouse-v0"  # [RobotWarehouse-v0, MultiCVRP-v0]
    ACTIVATION = "relu"
    UPDATE_EPOCHS = 1
    NUM_MINIBATCHES = 1
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    BATCH_SIZE = 16  # Parallel updates / environmnents
    ROLLOUT_LENGTH = 128  # Length of each rollout
    TOTAL_TIMESTEPS = 1000000  # Number of training timesteps
    SEED = 42
    # Logging config
    USE_TF = True
    USE_SACRED = True


@ex.main
def run_experiment(_run, _config, _log):
    # Logger setup
    config = config_copy(_config)
    logger = Logger(_log)
    unique_token = (
        f"{_config['ENV_NAME']}_seed{_config['SEED']}_{datetime.datetime.now()}"
    )
    if config["USE_SACRED"]:
        logger.setup_sacred(_run)
    if config["USE_TF"]:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)

    # Environment setup
    if "MultiCVRP" in config["ENV_NAME"]:
        config["NUM_AGENTS"] = 3
        generator = UniformRandomGenerator(
            num_vehicles=config["NUM_AGENTS"], num_customers=6
        )
        env = jumanji.make(config["ENV_NAME"], generator=generator)
        num_actions = int(env.action_spec().maximum)

    elif "RobotWarehouse" in config["ENV_NAME"]:
        env = jumanji.make(config["ENV_NAME"])
        num_actions = int(env.action_spec().num_values[0])
        config["NUM_AGENTS"] = env.num_agents
    env = AutoResetWrapper(env)
    env = LogWrapper(env)
    cores_count = len(jax.devices())
    # Number of iterations
    timesteps_per_iteration = (
        cores_count * config["ROLLOUT_LENGTH"] * config["BATCH_SIZE"]
    )
    config["ITERATIONS"] = (
        config["TOTAL_TIMESTEPS"] // timesteps_per_iteration
    )  # Number of training updates

    num_frames = config["TOTAL_TIMESTEPS"]

    network = ActorCritic(
        num_actions, activation=config["ACTIVATION"], env_name=config["ENV_NAME"]
    )
    optim = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )

    rng = jax.random.PRNGKey(config["SEED"])
    rng, rng_env, rng_params = jax.random.split(rng, 3)

    init_obs = env.observation_spec().generate_value()
    init_obs = jax.tree_util.tree_map(
        lambda x: x[None, ...],
        init_obs,
    )

    params = network.init(rng_params, init_obs)
    opt_state = optim.init(params)

    # TODO: Complete this
    learn = get_learner_fn(env, network.apply, optim.update, config)

    learn = jax.pmap(learn, axis_name="i")  # replicate over multiple cores.

    broadcast = lambda x: jnp.broadcast_to(
        x, (cores_count, config["BATCH_SIZE"]) + x.shape
    )
    params = jax.tree_map(broadcast, params)  # broadcast to cores and batch.
    opt_state = jax.tree_map(broadcast, opt_state)  # broadcast to cores and batch

    rng, *env_rngs = jax.random.split(rng, cores_count * config["BATCH_SIZE"] + 1)
    env_states, env_timesteps = jax.vmap(env.reset)(jnp.stack(env_rngs))  # init envs.
    rng, *step_rngs = jax.random.split(rng, cores_count * config["BATCH_SIZE"] + 1)

    reshape = lambda x: x.reshape((cores_count, config["BATCH_SIZE"]) + x.shape[1:])
    step_rngs = reshape(jnp.stack(step_rngs))  # add dimension to pmap over.
    env_states = jax.tree_util.tree_map(
        reshape,
        env_states,
    )  # add dimension to pmap over.
    env_timesteps = jax.tree_util.tree_map(
        reshape,
        env_timesteps,
    )
    # env_timesteps = reshape(env_timesteps)  # add dimension to pmap over.

    with TimeIt(tag="COMPILATION"):
        out = learn(params, opt_state, step_rngs, env_states, env_timesteps)  # compiles
        jax.block_until_ready(out)

    with TimeIt(tag="EXECUTION", frames=num_frames):
        out = learn(  # runs compiled fn
            params,
            opt_state,
            step_rngs,
            env_states,
            env_timesteps,
        )
        jax.block_until_ready(out)
    log(logger, out["metrics"])


if __name__ == "__main__":
    file_obs_path = os.path.join(
        results_path, f"sacred/"
    )  # TODO: Change the path to include the env and scenario names.
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run()
