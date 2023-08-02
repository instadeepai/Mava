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

from typing import Any, Callable, Dict, Sequence, Tuple

import chex
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
import jumanji
import numpy as np
import optax
from flax.core.frozen_dict import FrozenDict
from flax.linen.initializers import constant, orthogonal
from jumanji.env import Environment
from jumanji.types import Observation
from jumanji.wrappers import AutoResetWrapper
from optax._src.base import OptState

from jax_distribution.types import ExperimentOutput, PPOTransition, RunnerState
from jax_distribution.utils.jax import merge_leading_dims
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
        actor_policy = distrax.Categorical(logits=masked_logits)

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

        return actor_policy, jnp.squeeze(critic_output, axis=-1)


def get_learner_fn(
    env: jumanji.Environment, apply_fn: Callable, update_fn: Callable, config: Dict
) -> Callable:
    """Get the learner function."""

    def _update_step(
        runner_state: RunnerState, _: Any
    ) -> Tuple[RunnerState, Dict[str, chex.Array]]:
        """A single update of the network.

        This function steps the environment and records the trajectory batch for
        training. It then calculates advantages and targets based on the recorded
        trajectory and updates the actor and critic networks based on the calculated
        losses.

        Args:
            runner_state (NamedTuple):
                - params (FrozenDict): The current model parameters.
                - opt_state (OptState): The current optimizer state.
                - rng (PRNGKey): The random number generator state.
                - env_state (State): The environment state.
                - last_timestep (TimeStep): The last timestep in the current trajectory.
            _ (Any): The current metrics info.
        """

        def _env_step(
            runner_state: RunnerState, _: Any
        ) -> Tuple[RunnerState, PPOTransition]:
            """Step the environment."""
            params, opt_state, rng, env_state, last_timestep = runner_state

            # SELECT ACTION
            rng, policy_rng = jax.random.split(rng)
            actor_policy, value = apply_fn(params, last_timestep.observation)
            action = actor_policy.sample(seed=policy_rng)
            log_prob = actor_policy.log_prob(action)

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
                "episode_return_info": env_state.episode_return_info,
                "episode_length_info": env_state.episode_length_info,
            }

            transition = PPOTransition(
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
            traj_batch: PPOTransition, last_val: chex.Array
        ) -> Tuple[chex.Array, chex.Array]:
            """Calculate the GAE."""

            def _get_advantages(
                gae_and_next_value: Tuple, transition: PPOTransition
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

        def _update_epoch(update_state: Tuple, _: Any) -> Tuple:
            """Update the network for a single epoch."""

            def _update_minibatch(train_state: Tuple, batch_info: Tuple) -> Tuple:
                """Update the network for a single minibatch."""
                params, opt_state = train_state
                traj_batch, advantages, targets = batch_info

                def _loss_fn(
                    params: FrozenDict,
                    opt_state: OptState,
                    traj_batch: PPOTransition,
                    gae: chex.Array,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the loss."""
                    # RERUN NETWORK
                    actor_policy, value = apply_fn(params, traj_batch.obs)
                    log_prob = actor_policy.log_prob(traj_batch.action)

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
                    entropy = actor_policy.entropy().mean()

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

                # Compute the parallel mean (pmean) over the batch.
                # This calculation is inspired by the Anakin architecture demo.
                # This pmean could be a regular mean as the batch axis is on all devices.
                grads, total_loss = jax.lax.pmean(
                    (grads, total_loss), axis_name="batch"
                )
                # pmean over devices.
                grads, total_loss = jax.lax.pmean(
                    (grads, total_loss), axis_name="device"
                )

                updates, new_opt_state = update_fn(grads, opt_state)
                new_params = optax.apply_updates(params, updates)

                return (new_params, new_opt_state), total_loss

            params, opt_state, traj_batch, advantages, targets, rng = update_state
            rng, shuffle_rng = jax.random.split(rng)

            # SHUFFLE MINIBATCHES
            batch_size = config["ROLLOUT_LENGTH"] * config["NUM_ENVS"]
            permutation = jax.random.permutation(shuffle_rng, batch_size)
            batch = (traj_batch, advantages, targets)
            batch = jax.tree_util.tree_map(lambda x: merge_leading_dims(x, 2), batch)
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

    def learner_fn(runner_state: RunnerState) -> ExperimentOutput:
        """Learner function.

        This function represents the learner, it updates the network parameters
        by iteratively applying the `_update_step` function for a fixed number of
        updates. The `_update_step` function is vectorized over a batch of inputs.

        Args:
            runner_state (NamedTuple):
                - params (FrozenDict): The initial model parameters.
                - opt_state (OptState): The initial optimizer state.
                - rng (chex.PRNGKey): The random number generator state.
                - env_state (LogEnvState): The environment state.
                - timesteps (TimeStep): The initial timestep in the initial trajectory.
        """

        batched_update_step = jax.vmap(
            _update_step, in_axes=(0, None), axis_name="batch"
        )

        runner_state, metric = jax.lax.scan(
            batched_update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return learner_fn


def learner_setup(env: Environment, config: Dict) -> Tuple[callable, RunnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())
    # Get number of actions and agents.
    num_actions = int(env.action_spec().num_values[0])
    num_agents = env.action_spec().shape[0]
    config["NUM_AGENTS"] = num_agents
    # PRNG keys.
    rng, rng_p = jax.random.split(jax.random.PRNGKey(config["SEED"]))

    # Define network and optimiser.
    network = ActorCritic(num_actions, config["ACTIVATION"])
    optim = optax.adam(config["LR"])

    # Initialise observation: Select only obs for a single agent.
    init_x = env.observation_spec().generate_value()
    init_x = jax.tree_util.tree_map(lambda x: x[0], init_x)
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)

    # initialise params and optimiser state.
    params = network.init(rng_p, init_x)
    opt_state = optim.init(params)

    # Vmap network apply function over number of agents.
    vmapped_network_apply_fn = jax.vmap(
        network.apply, in_axes=(None, 1), out_axes=(1, 1)
    )

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, vmapped_network_apply_fn, optim.update, config)
    learn = jax.pmap(learn, axis_name="device")

    # Broadcast params and optimiser state to cores and batch.
    broadcast = lambda x: jnp.broadcast_to(
        x, (n_devices, config["UPDATE_BATCH_SIZE"]) + x.shape
    )
    params = jax.tree_map(broadcast, params)
    opt_state = jax.tree_map(broadcast, opt_state)

    # Initialise environment states and timesteps.
    rng, *env_rngs = jax.random.split(
        rng, n_devices * config["UPDATE_BATCH_SIZE"] * config["NUM_ENVS"] + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_rngs),
    )

    # Split rngs for each core.
    rng, *step_rngs = jax.random.split(rng, n_devices * config["UPDATE_BATCH_SIZE"] + 1)
    # Add dimension to pmap over.
    reshape_step_rngs = lambda x: x.reshape(
        (n_devices, config["UPDATE_BATCH_SIZE"]) + x.shape[1:]
    )
    step_rngs = reshape_step_rngs(jnp.stack(step_rngs))
    reshape_states = lambda x: x.reshape(
        (n_devices, config["UPDATE_BATCH_SIZE"], config["NUM_ENVS"]) + x.shape[1:]
    )
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    return learn, (params, opt_state, step_rngs, env_states, timesteps)


def run_experiment(env: Environment, config: Dict) -> ExperimentOutput:
    """Runs experiment."""
    # Setup learner.
    learn, initial_runner_state = learner_setup(env, config)

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    total_timesteps = (
        n_devices
        * config["NUM_UPDATES"]
        * config["ROLLOUT_LENGTH"]
        * config["UPDATE_BATCH_SIZE"]
        * config["NUM_ENVS"]
    )

    # Run experiment.
    with TimeIt(tag="COMPILATION"):
        learn(initial_runner_state)

    with TimeIt(tag="EXECUTION", environment_steps=total_timesteps):
        learner_output = learn(initial_runner_state)

    return learner_output


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

    learner_output = run_experiment(env, config)
    print("IPPO experiment completed")
