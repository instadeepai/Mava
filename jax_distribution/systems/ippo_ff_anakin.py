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

import timeit
from typing import Any, NamedTuple, Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jumanji
import numpy as np
import optax
from flax import struct
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import constant, orthogonal
from jumanji.environments.routing.multi_cvrp.generator import UniformRandomGenerator
from jumanji.environments.routing.multi_cvrp.types import State
from jumanji.types import TimeStep
from jumanji.wrappers import AutoResetWrapper, Wrapper
from flax.training.train_state import TrainState


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
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        state = LogEnvState(state, 0.0, 0, 0.0, 0)
        return state, timestep

    def step(
        self,
        state: State,
        action: jnp.ndarray,
    ) -> Tuple[State, TimeStep]:
        """Step the environment."""
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

    def get_num_agents(self) -> int:
        """Get the number of agents in the environment"""
        if "MultiCVRP" in self._env_name:
            num_agents = self._env._num_vehicles
        elif "RobotWarehouse" in self._env_name:
            num_agents = self._env.num_agents
        else:
            raise NotImplementedError("This environment is not supported")
        return num_agents

    def get_num_actions(self) -> int:
        """Get the number of actions in the environment"""
        if "MultiCVRP" in self._env_name:
            num_actions = int(self._env.action_spec().maximum)
        elif "RobotWarehouse" in self._env_name:
            num_actions = int(self._env.action_spec().num_values[0])
        else:
            raise NotImplementedError("This environment is not supported")
        return num_actions

def process_observation(observation: Any, env_name: str) -> Any:
    """Process the observation to be fed into the network based on the environment."""
    # TODO: We probably need an preprocessing function here
    # If else is maybe not the best solution going forward.
    if "MultiCVRP" in env_name:
        observation = observation.vehicles.coordinates
    elif "RobotWarehouse" in env_name:
        observation = observation.agents_view
    else:
        raise NotImplementedError("This environment is not supported")
    return observation

def get_env(env_name: str) -> jumanji.Environment:
    """Create the environment."""
    if "MultiCVRP" in env_name:
        if num_agents is None:
            num_agents = 3
        generator = UniformRandomGenerator(num_vehicles=num_agents, num_customers=6)
        env = jumanji.make(env_name, generator=generator)
        num_actions = int(env.action_spec().maximum)
    elif "RobotWarehouse" in env_name:
        env = jumanji.make(env_name)
        num_actions = int(env.action_spec().num_values[0])
        num_agents = env.num_agents
    return env, num_agents, num_actions

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    observation: jnp.ndarray
    info: jnp.ndarray


class ActorCritic(nn.Module):
    """Actor Critic Network."""
    action_dim: Sequence[int]
    env_name: str
    activation: str = "tanh"

    @nn.compact
    def __call__(self, observation) -> Tuple[distrax.Categorical, jnp.ndarray]:

        x = process_observation(observation, self.env_name)

        """Forward pass."""
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_output = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)

        actor_output = activation(actor_output)
        actor_output = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(actor_output)
        actor_output = activation(actor_output)

        actor_output = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_output)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_output,
            jnp.finfo(jnp.float32).min,
        )

        pi = distrax.Categorical(logits=masked_logits)

        critic_output = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        critic_output = activation(critic_output)
        critic_output = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(critic_output)
        critic_output = activation(critic_output)
        critic_output = nn.Dense(
            1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)
        )(critic_output)

        return pi, jnp.squeeze(critic_output, axis=-1)


def get_runner_fn(
    env: jumanji.Environment,
    network: nn.Module,
    config: dict,
) -> callable:
    """Get the learner function."""

    def _update_step(runner_state, unused) -> Tuple:
        # COLLECT TRAJECTORIES
        def _env_step(runner_state: Tuple, unused: Any) -> Tuple:
            """Step the environment."""
            train_state, env_state, last_observation, rng = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = network.apply(train_state.params, last_observation)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENV
            rng, _rng = jax.random.split(rng)

            env_state, next_timestep = env.step(env_state, action)

            done, reward, ep_returns, ep_lengths, ep_done = jax.tree_map(
                lambda x: jnp.repeat(x, config["NUM_AGENTS"]).reshape(-1),
                [next_timestep.last(),
                 next_timestep.reward,
                 env_state.returned_episode_returns,
                 env_state.returned_episode_lengths,
                 next_timestep.last()],
            )

            observation = next_timestep.observation
            transition = Transition(
                done,
                action,
                value,
                reward,
                log_prob,
                observation,
                {
                    "returned_episode_returns": ep_returns,
                    "returned_episode_lengths": ep_lengths,
                    "returned_episode": ep_done,
                },
            )

            
            runner_state = (train_state, env_state, observation, rng)
            return runner_state, transition

        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
        )

        # CALCULATE ADVANTAGE
        train_state, env_state, last_observation, rng = runner_state
        _, last_val = network.apply(train_state.params, last_observation)

        def _calculate_gae(
            traj_batch: Transition, last_val: jnp.ndarray
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Calculate the GAE."""

            def _get_advantages(
                gae_and_next_value: Tuple, transition: Transition
            ) -> Tuple:
                """Calculate the GAE for a single transition."""
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
        def _update_epoch(update_state: Tuple, unused: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minbatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                traj_batch, advantages, targets = batch_info

                def _loss_fn(
                    params: FrozenDict,
                    traj_batch: Transition,
                    gae: jnp.ndarray,
                    targets: jnp.ndarray,
                ) -> Tuple:
                    """Calculate the loss."""
                    # RERUN NETWORK
                    pi, value = network.apply(params, traj_batch.observation)
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
                total_loss, grads = grad_fn(train_state.params, traj_batch, advantages, targets)

                # pmean
                total_loss = jax.lax.pmean(total_loss, axis_name="device")
                grads = jax.lax.pmean(grads, axis_name="batch")
                grads = jax.lax.pmean(grads, axis_name="device")
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            train_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)
            batch_size = config["ROLLOUT_LENGTH"]
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


        runner_state = (train_state, env_state, last_observation, rng)
        return runner_state, metric

    def runner_fn(runner_state):
        """Vectorise and repeat the update."""
        batched_update_fn = jax.vmap(
            _update_step, axis_name="batch"
        )  # vectorize across batch.

        runner_state, metric = jax.lax.scan(
        batched_update_fn, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}
       
    return runner_fn


def run_experiment(env_name, config):
    """Run the experiment.
    
    Args:
        env_name: Name of the environment.
        config: Configuration dictionary.
        
    Returns:
        None
    """
    env, config["NUM_AGENTS"], num_actions = get_env(env_name)
    env= AutoResetWrapper(env)
    env = LogWrapper(env)

    cores_count = len(jax.devices())

    # Number of updates
    timesteps_per_update = (
        cores_count * config["ROLLOUT_LENGTH"] * config["BATCH_SIZE"]
    )
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // timesteps_per_update
    )  # Number of training updates
    total_time_steps = config["TOTAL_TIMESTEPS"]

    # Create the network
    network = ActorCritic(
        num_actions,
        env_name=env_name,
        activation=config["ACTIVATION"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    init_obs = env.observation_spec().generate_value()

    rng, _rng = jax.random.split(rng)
    network_params = network.init(_rng, init_obs)
    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(config["LR"], eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    # Create the runner function.
    runner = get_runner_fn(
        env,
        network,
        config,
    )
    runner = jax.pmap(runner, axis_name="device")  # replicate over multiple cores.

    # BROADCAST TRAIN STATE
    def broadcast(x):
        if np.isscalar(x):  # x is an int or float
            x = jnp.array(x)  # convert it to a JAX array
        return jnp.broadcast_to(x, (cores_count, config["BATCH_SIZE"]) + x.shape)
    train_state = jax.tree_map(broadcast, train_state) # broadcast to cores and batch.

    rng, *reset_rngs = jax.random.split(rng, 1 + cores_count * config["BATCH_SIZE"] )
    env_states, env_timesteps = jax.vmap(env.reset)(jnp.stack(reset_rngs))  # init envs.
   
    # RESHAPE OBSERVATION, ENV STATES AND STEP RNGS
    reshape = lambda x: x.reshape((cores_count, config["BATCH_SIZE"]) + x.shape[1:])
    rng, *step_rngs = jax.random.split(rng, 1 + cores_count * config["BATCH_SIZE"])
    env_states, observation, step_rngs = jax.tree_util.tree_map(
        reshape,
        [env_states, env_timesteps.observation, jnp.array(step_rngs)],
    )  # add dimension to pmap over.
    runner_state = (train_state, env_states, observation, step_rngs)

    # Run the experiment.
    with TimeIt(tag="COMPILATION"):
        out = runner(runner_state)  # compiles
        jax.block_until_ready(out)

    with TimeIt(tag="EXECUTION", frames=total_time_steps):
        out = runner(runner_state) # runs compiled fn
        jax.block_until_ready(out)
    val = out["metric"]["returned_episode_returns"].mean()
    print(f"Mean Episode Return: {val}")


if __name__ == "__main__":
    config = {
        "LR": 5e-3,
        "ENV_NAME": "RobotWarehouse-v0",  # [RobotWarehouse-v0, MultiCVRP-v0]
        "ACTIVATION": "relu",
        "UPDATE_EPOCHS": 1,
        "NUM_MINIBATCHES": 1,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "BATCH_SIZE": 4,  # Parallel updates / environmnents
        "ROLLOUT_LENGTH": 128,  # Length of each rollout
        "TOTAL_TIMESTEPS": 204800,  # Number of training timesteps
        "SEED": 42,
    }

    run_experiment(config["ENV_NAME"], config)
