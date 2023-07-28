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

from typing import Any, Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jumanji
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from jumanji.env import Environment
from jumanji.environments.routing.robot_warehouse import Observation
from jumanji.wrappers import AutoResetWrapper

from jax_distribution.types import Transition
from jax_distribution.utils.timing_utils import TimeIt
from jax_distribution.wrappers.jumanji import LogWrapper, RwareMultiAgentWrapper


class ActorCritic(nn.Module):
    """Actor Critic Network."""

    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(
        self, observation: Observation
    ) -> Tuple[distrax.Categorical, chex.Array]:
        """Forward pass."""
        x = observation.agents_view
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


def get_learner_fn(
    env: jumanji.Environment, apply_fn: callable, update_fn: callable, config: dict
) -> callable:
    """Get the learner function."""

    def _update_step(runner_state: Tuple, unused_target: Any) -> Tuple:
        """Update the network."""

        def _env_step(runner_state: Tuple, unused: Any) -> Tuple:
            """Step the environment."""
            params, opt_state, rng, env_state, last_timestep = runner_state

            # SELECT ACTION
            rng, _rng = jax.random.split(rng)
            pi, value = apply_fn(params, last_timestep.observation)
            action = pi.sample(seed=_rng)
            log_prob = pi.log_prob(action)

            # STEP ENVIRONMENT
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # LOG EPISODE METRICS
            done, reward = jax.tree_util.tree_map(
                lambda x: jnp.repeat(x, config["NUM_AGENTS"]).reshape(
                    config["NUM_ENVS"], -1
                ),
                (timestep.last(), timestep.reward),
            )
            info = {
                "returned_episode_returns": env_state.returned_episode_returns,
                "returned_episode_lengths": env_state.returned_episode_lengths,
            }

            transition = Transition(
                done, action, value, reward, log_prob, last_timestep.observation, info
            )
            runner_state = (params, opt_state, rng, env_state, timestep)
            return runner_state, transition

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
        )

        # CALCULATE ADVANTAGE
        params, opt_state, rng, env_state, last_timestep = runner_state
        _, last_val = apply_fn(params, last_timestep.observation)

        def _calculate_gae(
            traj_batch: Transition, last_val: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
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

        def _update_epoch(update_state: Tuple, unused: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                params, opt_state = train_state
                traj_batch, advantages, targets = batch_info

                def _loss_fn(
                    params,
                    opt_state,
                    traj_batch: Transition,
                    gae: chex.Array,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the loss."""
                    # RERUN NETWORK
                    pi, value = apply_fn(params, traj_batch.obs)
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
                    params, opt_state, traj_batch, advantages, targets
                )

                grads, total_loss = jax.lax.pmean(
                    (grads, total_loss), axis_name="batch"
                )
                grads, total_loss = jax.lax.pmean(
                    (grads, total_loss), axis_name="device"
                )

                updates, new_opt_state = update_fn(grads, opt_state)
                new_params = optax.apply_updates(params, updates)

                return (new_params, new_opt_state), total_loss

            params, opt_state, traj_batch, advantages, targets, rng = update_state
            rng, _rng = jax.random.split(rng)

            # SHUFFLE MINIBATCHES
            batch_size = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
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

            # UPDATE MINIBATCHES
            (params, opt_state), total_loss = jax.lax.scan(
                _update_minibatch, (params, opt_state), minibatches
            )

            update_state = (params, opt_state, traj_batch, advantages, targets, rng)
            return update_state, total_loss

        update_state = (params, opt_state, traj_batch, advantages, targets, rng)

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["PPO_EPOCHS"]
        )

        params, opt_state, traj_batch, advantages, targets, rng = update_state
        runner_state = (params, opt_state, rng, env_state, last_timestep)
        metric = traj_batch.info
        return runner_state, metric

    def learner_fn(params, opt_state, rng, env_state, timesteps) -> dict:
        """Learner function."""
        runner_state = (params, opt_state, rng, env_state, timesteps)

        batched_update_step = jax.vmap(
            _update_step, in_axes=(0, None), axis_name="batch"
        )

        runner_state, metric = jax.lax.scan(
            batched_update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return learner_fn


def run_experiment(env: Environment, config: dict) -> dict:
    """Runs experiment."""
    # INITIALISE
    cores_count = len(jax.devices())  # get available TPU cores.
    num_actions = int(env.action_spec().num_values[0])
    num_agents = env.action_spec().shape[0]
    config["NUM_AGENTS"] = num_agents
    network = ActorCritic(num_actions, config["ACTIVATION"])  # define network.
    optim = optax.adam(config["LR"])  # define optimiser.
    rng, rng_e, rng_p = jax.random.split(
        jax.random.PRNGKey(config["SEED"]), num=3
    )  # prng keys.
    total_timesteps = (
        cores_count
        * config["NUM_UPDATES"]
        * config["ROLLOUT_LENGTH"]
        * config["UPDATE_BATCH_SIZE"]
        * config["NUM_ENVS"]
    )

    # INITIALISE NETWORK AND OPTIMISER PARAMS
    init_x = env.observation_spec().generate_value()
    # select only obs for a single agent.
    init_x = jax.tree_util.tree_map(lambda x: x[0], init_x)
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)
    params = network.init(rng_p, init_x)
    # initialise optimiser stats.
    opt_state = optim.init(params)

    vmapped_network_apply_fn = jax.vmap(
        network.apply, in_axes=(None, 1), out_axes=(1, 1)
    )

    learn = get_learner_fn(  # get batched iterated update.
        env, vmapped_network_apply_fn, optim.update, config
    )
    learn = jax.pmap(learn, axis_name="device")  # replicate over multiple cores.

    # BROADCAST PARAMS AND OPTIMISER STATE
    broadcast = lambda x: jnp.broadcast_to(
        x, (cores_count, config["UPDATE_BATCH_SIZE"]) + x.shape
    )
    params = jax.tree_map(broadcast, params)  # broadcast to cores and batch.
    opt_state = jax.tree_map(broadcast, opt_state)  # broadcast to cores and batch

    # INITIALISE ENVIRONMENT
    rng, *env_rngs = jax.random.split(
        rng, cores_count * config["UPDATE_BATCH_SIZE"] * config["NUM_ENVS"] + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_rngs),
    )
    rng, *step_rngs = jax.random.split(
        rng, cores_count * config["UPDATE_BATCH_SIZE"] + 1
    )

    # RESHAPE ENVIRONMENT STATES AND TIMESTEPS
    reshape_step_rngs = lambda x: x.reshape(
        (cores_count, config["UPDATE_BATCH_SIZE"]) + x.shape[1:]
    )
    step_rngs = reshape_step_rngs(jnp.stack(step_rngs))  # add dimension to pmap over.

    reshape_states = lambda x: x.reshape(
        (cores_count, config["UPDATE_BATCH_SIZE"], config["NUM_ENVS"]) + x.shape[1:]
    )
    env_states = jax.tree_util.tree_map(
        reshape_states, env_states
    )  # add dimension to pmap over.
    timesteps = jax.tree_util.tree_map(
        reshape_states, timesteps
    )  # add dimension to pmap over.

    # RUN EXPERIMENT
    with TimeIt(tag="COMPILATION"):
        learn(params, opt_state, step_rngs, env_states, timesteps)

    with TimeIt(tag="EXECUTION", environment_steps=total_timesteps):
        output = learn(params, opt_state, step_rngs, env_states, timesteps)

    return output


if __name__ == "__main__":
    config = {
        "LR": 2.5e-4,
        "UPDATE_BATCH_SIZE": 4,
        "ROLLOUT_LENGTH": 128,
        "NUM_UPDATES": 10,
        "NUM_ENVS": 32,
        "PPO_EPOCHS": 4,
        "NUM_MINIBATCHES": 8,
        "GAMMA": 0.99,
        "GAE_LAMBDA": 0.95,
        "CLIP_EPS": 0.2,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "ACTIVATION": "relu",
        "ENV_NAME": "RobotWarehouse-v0",
        "SEED": 42,
    }

    env = jumanji.make(config["ENV_NAME"])
    env = RwareMultiAgentWrapper(env)
    env = AutoResetWrapper(env)
    env = LogWrapper(env)

    output = run_experiment(env, config)
