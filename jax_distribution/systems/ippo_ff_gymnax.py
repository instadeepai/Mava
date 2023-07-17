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
from typing import Any, NamedTuple, Optional, Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
import gymnax
from jax_distribution.wrappers.purejaxrl import FlattenObservationWrapper, LogWrapper
import matplotlib.pyplot as plt

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

def get_env(env_name: str, num_agents: Optional[int] = None):
    """Create the environment."""
    env, env_params = gymnax.make(config["ENV_NAME"])
    
    # Set the reset and steps
    env._main_reset = lambda rng: env.reset(rng, env_params)
    env._main_step = lambda rng, env_state, action: env.step(rng, env_state, action, env_params)
    num_agents = 1
    num_actions = env.action_space().n
    return env, num_agents, num_actions, env_params


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    observation: jnp.ndarray
    info: jnp.ndarray


def plot_episode_returns(episode_returns, timesteps):
    plt.figure(figsize=(10,5))
    plt.plot(timesteps, episode_returns)
    plt.xlabel('Timesteps')
    plt.ylabel('Episode Returns')
    plt.title('Episode Returns over Timesteps')
    plt.grid(True)
    plt.show()


class ActorCritic(nn.Module):
    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
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
        pi = distrax.Categorical(logits=actor_mean)

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


def get_runner_fn(
    env,
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

            obs, env_state, reward, done, info = env.step(_rng, env_state, action)
            
            # This is just for the CartPole-v1 environment
            # TODO: Remove action and value from here.
            assert config["NUM_AGENTS"] == 1
            done, reward, action, value, log_prob, observation = jax.tree_map(
                lambda x: jnp.repeat(x, config["NUM_AGENTS"]).reshape(-1),
                [
                    done,
                    reward,
                    action,
                    value,
                    log_prob,
                    obs,
                ],
            )

            transition = Transition(
                done,
                action,
                value,
                reward,
                log_prob,
                observation,
                {
                    "returned_episode_returns": info["returned_episode_returns"],
                    "returned_episode_lengths": info["returned_episode_lengths"],
                    "returned_episode": done,
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

        # TODO: Remove this
        last_val = last_val.reshape((-1,))

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
                total_loss, grads = grad_fn(
                    train_state.params, traj_batch, advantages, targets
                )

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


def run_experiment(env_name: str, config: dict) -> None:
    """Run the experiment.

    Args:
        env_name: Name of the environment.
        config: Configuration dictionary.

    Returns:
        None
    """
    env, config["NUM_AGENTS"], num_actions, env_params = get_env(env_name)
    env = LogWrapper(env)

    cores_count = len(jax.devices())

    # Number of updates
    timesteps_per_update = cores_count * config["ROLLOUT_LENGTH"] * config["BATCH_SIZE"]
    config["NUM_UPDATES"] = int(
        config["TOTAL_TIMESTEPS"] // timesteps_per_update
    )  # Number of training updates
    total_time_steps = config["TOTAL_TIMESTEPS"]

    # Create the network
    network = ActorCritic(
        num_actions,
        activation=config["ACTIVATION"],
    )

    rng = jax.random.PRNGKey(config["SEED"])
    init_obs = jnp.zeros(env.observation_space(env_params).shape)

    rng, _rng = jax.random.split(rng)
    network_params = network.init(_rng, init_obs)
    
    def linear_schedule(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return config["LR"] * frac

    tx = optax.chain(
        optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
        optax.adam(learning_rate=linear_schedule, eps=1e-5),
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

    train_state = jax.tree_map(broadcast, train_state)  # broadcast to cores and batch.

    rng, *reset_rngs = jax.random.split(rng, 1 + cores_count * config["BATCH_SIZE"])
    env_timesteps, env_states = jax.vmap(env.reset)(jnp.stack(reset_rngs))  # init envs.

    # RESHAPE OBSERVATION, ENV STATES AND STEP RNGS
    reshape = lambda x: x.reshape((cores_count, config["BATCH_SIZE"]) + x.shape[1:])
    rng, *step_rngs = jax.random.split(rng, 1 + cores_count * config["BATCH_SIZE"])
    env_states, observation, step_rngs = jax.tree_util.tree_map(
        reshape,
        [env_states, env_timesteps, jnp.array(step_rngs)],
    )  # add dimension to pmap over.
    runner_state = (train_state, env_states, observation, step_rngs)

    # Run the experiment.
    with TimeIt(tag="COMPILATION"):
        out = runner(runner_state)  # compiles
        jax.block_until_ready(out)

    with TimeIt(tag="EXECUTION", frames=total_time_steps):
        out = runner(runner_state)  # runs compiled fn
        jax.block_until_ready(out)

    val = out["metric"]["returned_episode_returns"].mean()
    print(f"Mean Episode Return: {val}")
    val = out["metric"]["returned_episode_lengths"].mean()
    print(f"Mean Episode Length: {val}")
    ep_returns = out["metric"]["returned_episode_returns"].reshape((config["NUM_UPDATES"], -1)).mean(axis=1)

    plot_episode_returns(ep_returns, np.arange(0, config["NUM_UPDATES"]))

if __name__ == "__main__":
    config = {
        "LR": 5e-3,
        "ENV_NAME": "CartPole-v1",
        "ACTIVATION": "tanh",
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "BATCH_SIZE": 4,  # Parallel updates / environmnents
        "ROLLOUT_LENGTH": 128,  # Length of each rollout
        "TOTAL_TIMESTEPS": 5e5,  # Number of training timesteps
        "SEED": 42,
    }

    run_experiment(config["ENV_NAME"], config)
