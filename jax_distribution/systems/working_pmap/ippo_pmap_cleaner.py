# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any
from flax.training.train_state import TrainState
from flax import struct
import distrax
import jumanji
from jumanji.wrappers import AutoResetWrapper
# from jax_distribution.wrappers.purejaxrl import LogWrapper, FlattenObservationWrapper
import time
from jumanji.environments.routing.robot_warehouse.types import State
from jumanji.wrappers import Wrapper
from jumanji.types import TimeStep
import chex 
from typing import Tuple
import timeit

#TODO: import these from jumanji 
DIRTY = 0
CLEAN = 1
WALL = 2

class TimeIt:
    """Context manager for timing execution."""

    def __init__(self, tag: str, frames=None) -> None:
        """Initialise the context manager."""
        self.tag = tag
        self.frames = frames

    def __enter__(self) -> "TimeIt":
        """Start the timer."""
        self.start = timeit.default_timer()
        return self

    def __exit__(self, *args: Any) -> None:
        """Print the elapsed time."""
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
        state = LogEnvState(state, 0.0, 0.0, 0.0, 0.0)
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

def all_agents_channel(agents_locations: chex.Array, grid: chex.Array) -> chex.Array:
    """Create a channel containing the number of agents per tile.

    Args:
        grid: the maze grid.
        agents_locations: the location of all the agents.

    Returns:
        array: a 2D jax array containing the number of agents on each tile."""

    xs, ys = agents_locations[:, 0], agents_locations[:, 1]
    num_agents = agents_locations.shape[0]
    agents_channel = jnp.repeat(jnp.zeros_like(grid)[None, :, :], num_agents, axis=0)
    return jnp.sum(agents_channel.at[jnp.arange(num_agents), xs, ys].set(1), axis=0)
    


def process_obs(observation) -> chex.Array:
    """Process the `Observation`.

    Args:
        observation: the observation as returned by the environment.

    Returns:
        array: a 4D jax array with 4 channels per agent:
            - Dirty channel: 2D array with 1 for dirty tiles and 0 otherwise.
            - Wall channel: 2D array with 1 for walls and 0 otherwise.
            - Agent channel: 2D array with 1 for the agent position and 0 otherwise.
            - Agents channel: 2D array with the number of agents on each tile.
    """
    grid = observation.grid
    agents_locations = observation.agents_locations

    def create_channels_for_one_agent(agent_location: chex.Array) -> chex.Array:
        dirty_channel = jnp.where(grid == DIRTY, 1, 0)
        wall_channel = jnp.where(grid == WALL, 1, 0)
        agent_channel = (
            jnp.zeros_like(grid).at[agent_location[0], agent_location[1]].set(1)
        )
        agents_channel = all_agents_channel(agents_locations, grid)
        return jnp.stack(
            [dirty_channel, wall_channel, agent_channel, agents_channel],
            axis=-1,
            dtype=float,
        )

    return jax.vmap(create_channels_for_one_agent)(agents_locations)
    
class Flatten(nn.Module):
    def __call__(self, x):
        return jnp.reshape(x, (x.shape[0],x.shape[1], -1))

class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, observation):
        
        # TODO: Include grid embedding to the network. 
        # Something like this. 
        # observation.grid (B, 10, 10) -> CNN -> (B, 1, H) -> (B, 3, H)
        # observation.agents_locations (B, 3, 2)

        # x = concat(grid_embedding, agent_locations) (B, 3, H+2)
        
        num_conv_channels=[4, 4, 1]
        conv_layers = [
            [
                nn.Conv(output_channels, (3, 3)),
                jax.nn.relu,
            ]
            for output_channels in num_conv_channels
        ]
        torso = nn.Sequential(
            [
                *[layer for conv_layer in conv_layers for layer in conv_layer],
                Flatten()
            ]
        )
        obs =jax.vmap(process_obs)(observation)
        embedding=torso(obs) # (B, N, W*H)
        x = embedding

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


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: dict


def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env = jumanji.make(config["ENV_NAME"])
    env = AutoResetWrapper(env)
    env = LogWrapper(env)

    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    def train(rng):
        # INIT NETWORK
        network = ActorCritic(
            4, activation=config["ACTIVATION"]
        )
        rng, _rng = jax.random.split(rng)
        init_x = env.observation_spec().generate_value()
        init_x = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), init_x)
        # init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)
        network_params = network.init(_rng, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(config["LR"], eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        env_state, timestep = jax.vmap(env.reset, in_axes=(0))(reset_rng)

        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                train_state, env_state, last_timestep, rng = runner_state

                # SELECT ACTION
                rng, _rng = jax.random.split(rng)
                pi, value = network.apply(train_state.params, last_timestep.observation)
                action = pi.sample(seed=_rng)
                log_prob = pi.log_prob(action)

                # TAKE ENV STEP
                env_state, timestep = jax.vmap(
                    env.step, in_axes=(0, 0)
                )(env_state, action)

                done, reward  = jax.tree_util.tree_map(
                    lambda x: jnp.repeat(x, 3).reshape(config["NUM_ENVS"], -1),
                    (timestep.last(), timestep.reward),
                )
                transition = Transition(
                    done, action, value, reward, log_prob, last_timestep.observation, 
                    {
                        "returned_episode_returns": env_state.returned_episode_returns,
                        "returned_episode_lengths": env_state.returned_episode_lengths,
                        "returned_episode": timestep.last(), 
                    }
                )
                runner_state = (train_state, env_state, timestep, rng)
                return runner_state, transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_timestep, rng = runner_state
            _, last_val = network.apply(train_state.params, last_timestep.observation)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done,
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1 - done) - value
                    gae = (
                        delta
                        + config["GAMMA"] * config["GAE_LAMBDA"] * (1 - done) * gae
                    )
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
                        pi, value = network.apply(params, traj_batch.obs)
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
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)
                batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                    batch_size == config["NUM_STEPS"] * config["NUM_ENVS"]
                ), "batch size must be equal to number of steps * number of envs"
                permutation = jax.random.permutation(_rng, batch_size)
                batch = (traj_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), batch
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
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )
                update_state = (train_state, traj_batch, advantages, targets, rng)
                return update_state, total_loss

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            metric = traj_batch.info
            rng = update_state[-1]
            if config.get("DEBUG"):
                def callback(info):
                    return_values = info["returned_episode_returns"][info["returned_episode"]]
                    timesteps = info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    for t in range(len(timesteps)):
                        print(f"global step={timesteps[t]}, episodic return={return_values[t]}")
                jax.debug.callback(callback, metric)

            # Pmean the train state
            types = jax.tree_util.tree_map(lambda x: x.dtype, train_state)
            train_state = jax.lax.pmean(train_state, axis_name="batch")
            train_state = jax.lax.pmean(train_state, axis_name="devices")
            train_state = jax.tree_util.tree_map(
                lambda x, t: jnp.asarray(x, dtype=t), train_state, types,
            ) 

            runner_state = (train_state, env_state, last_timestep, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (train_state, env_state, timestep, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return train


if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "NUM_ENVS": 512,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 2e6,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ENV_NAME": "Cleaner-v0",
        "ANNEAL_LR": True,
        "DEBUG": False,
    }
    rng = jax.random.PRNGKey(30)
    start = time.time()

    # Get the number of devices
    num_devices = jax.device_count()

    fn = make_train(config)
    
    # Num experiments to run
    num_exp = 32

    num_per_device = num_exp // num_devices
    assert num_exp == num_per_device * num_devices, "num_exp must be divisible by num_devices"

    # Vmap over experiments
    vmap_fn = jax.vmap(fn, in_axes=(0,), axis_name="batch")

    # Pmap over devices
    pmap_fn = jax.pmap(vmap_fn, in_axes=(0,), axis_name="devices")

    rngs = jax.random.split(rng, num_exp)

    # Reshape the keys
    rngs_reshaped = jnp.reshape(rngs, (num_devices, num_per_device, -1))

    with TimeIt(tag="COMPILATION"):
        print("Compiling")
        jax.block_until_ready(pmap_fn(rngs_reshaped))

    total_frames = config["TOTAL_TIMESTEPS"] * num_exp
    with TimeIt(tag="EXECUTION", frames=total_frames):
        print("Execution")
        out = pmap_fn(rngs_reshaped)
        jax.block_until_ready(out)

    val = out["metrics"]["returned_episode_returns"].mean()
    print(f"Mean Episode Return: {val}")
    val = out["metrics"]["returned_episode_lengths"].mean()
    print(f"Mean Episode Length: {val}")

    # Evaluate agent 
    print("STARTING EVAL")
    trained_params = jax.tree_util.tree_map(lambda x: x[0, 0, ...], out['runner_state'][0].params)
    network = ActorCritic(4, "relu")

    eval_env = jumanji.make(config["ENV_NAME"])
    eval_env = AutoResetWrapper(eval_env)

    rng = jax.random.PRNGKey(42)
    rng, _rng = jax.random.split(rng)
    init_obs = eval_env.observation_spec().generate_value()
    init_obs = jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), init_obs)
    _ = network.init(_rng, init_obs)


    states = []
    rng, _rng = jax.random.split(rng)
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    env_state, timestep = jit_reset(_rng)

    states.append(env_state)
    episode_returns = []
    current_return = 0 
    for _ in range(300): 

        obs=jax.tree_util.tree_map(lambda x: jnp.expand_dims(x, axis=0), timestep.observation)
        rng, _rng = jax.random.split(rng)
        pi, _ = network.apply(
            trained_params, 
            obs)
        
        action = pi.sample(seed=_rng)

        env_state, timestep = jit_step(env_state, action)
        if timestep.last():
            episode_returns.append(current_return)
            current_return = 0
        else:
            current_return += timestep.reward
        states.append(env_state)

    print(episode_returns)

    # eval_env.animate(states, interval=150, save_path="cleaner.gif")