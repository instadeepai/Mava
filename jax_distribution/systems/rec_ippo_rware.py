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

import datetime
import functools
import os
from logging import Logger as SacredLogger
from os.path import abspath, dirname
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
from sacred import Experiment
from sacred.observers import FileStorageObserver
from sacred.run import Run
from sacred.utils import apply_backspaces_and_linefeeds

from jax_distribution.types import ExperimentOutput, PPOTransition, RNNRunnerState
from jax_distribution.utils.jax import merge_leading_dims
from jax_distribution.utils.logger_tools import Logger, config_copy, get_logger
from jax_distribution.utils.timing_utils import TimeIt
from jax_distribution.wrappers.jumanji import LogWrapper, RwareMultiAgentWrapper


class ScannedRNN(nn.Module):
    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(
        self, carry: chex.Array, x: chex.Array
    ) -> Tuple[chex.Array, chex.Array]:
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell()(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size: int, hidden_size: int) -> chex.Array:
        """Initializes the carry state."""
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell.initialize_carry(
            jax.random.PRNGKey(0), (batch_size,), hidden_size
        )


class ActorCritic(nn.Module):
    """Actor Critic Network."""

    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(
        self, hidden: chex.Array, x: Observation
    ) -> Tuple[distrax.Categorical, chex.Array]:
        """Forward pass."""
        observation, dones = x
        obs = observation.agents_view

        embedding = nn.Dense(
            128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN()(hidden, rnn_in)

        actor_mean = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        masked_logits = jnp.where(
            observation.action_mask,
            actor_mean,
            jnp.finfo(jnp.float32).min,
        )

        pi = distrax.Categorical(logits=masked_logits)

        critic = nn.Dense(128, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )

        return hidden, pi, jnp.squeeze(critic, axis=-1)


def get_learner_fn(
    env: jumanji.Environment, apply_fn: Callable, update_fn: Callable, config: Dict
) -> Callable:
    """Get the learner function."""

    def _update_step(
        runner_state: RNNRunnerState, _: Any
    ) -> Tuple[RNNRunnerState, Dict[str, chex.Array]]:
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
                - last_done (bool): Whether the last timestep was a terminal state.
                - hstate (chex.Array): The hidden state of the RNN.
            _ (Any): The current metrics info.
        """

        def _env_step(
            runner_state: RNNRunnerState, _: Any
        ) -> Tuple[RNNRunnerState, PPOTransition]:
            """Step the environment."""
            (
                params,
                opt_state,
                rng,
                env_state,
                last_timestep,
                last_done,
                hstate,
            ) = runner_state

            rng, policy_rng = jax.random.split(rng)

            # Add a batch dimension to the observation.
            batched_observation = jax.tree_util.tree_map(
                lambda x: x[np.newaxis, :], last_timestep.observation
            )
            ac_in = (batched_observation, last_done[np.newaxis, :])

            # Run the network.
            hstate, actor_policy, value = apply_fn(params, hstate, ac_in)

            # Sample action from the policy and squeeze out the batch dimension.
            action = actor_policy.sample(seed=policy_rng)
            log_prob = actor_policy.log_prob(action)
            value, action, log_prob = (
                value.squeeze(0),
                action.squeeze(0),
                log_prob.squeeze(0),
            )

            # Step the environment.
            env_state, timestep = jax.vmap(env.step, in_axes=(0, 0))(env_state, action)

            # log episode return and length
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
            runner_state = (params, opt_state, rng, env_state, timestep, done, hstate)
            return runner_state, transition

        # INITIALISE RNN STATE
        initial_hstate = runner_state[-1]

        # STEP ENVIRONMENT FOR ROLLOUT LENGTH
        runner_state, traj_batch = jax.lax.scan(
            _env_step, runner_state, None, config["ROLLOUT_LENGTH"]
        )

        # CALCULATE ADVANTAGE
        (
            params,
            opt_state,
            rng,
            env_state,
            last_timestep,
            last_done,
            hstate,
        ) = runner_state

        # Add a batch dimension to the observation.
        batched_last_observation = jax.tree_util.tree_map(
            lambda x: x[np.newaxis, :], last_timestep.observation
        )
        ac_in = (batched_last_observation, last_done[np.newaxis, :])
        # Run the network.
        _, _, last_val = apply_fn(params, hstate, ac_in)
        # Squeeze out the batch dimension and mask out the value of terminal states.
        last_val = last_val.squeeze(0)
        last_val = jnp.where(last_done, jnp.zeros_like(last_val), last_val)

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
                init_hstate, traj_batch, advantages, targets = batch_info

                def _loss_fn(
                    params: FrozenDict,
                    opt_state: OptState,
                    traj_batch: PPOTransition,
                    gae: chex.Array,
                    targets: chex.Array,
                ) -> Tuple:
                    """Calculate the loss."""
                    # RERUN NETWORK
                    _, actor_policy, value = apply_fn(
                        params,
                        init_hstate.squeeze(0),
                        (traj_batch.obs, traj_batch.done),
                    )
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

            (
                params,
                opt_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state
            rng, shuffle_rng = jax.random.split(rng)

            # SHUFFLE MINIBATCHES
            permutation = jax.random.permutation(shuffle_rng, config["NUM_ENVS"])
            batch = (init_hstate, traj_batch, advantages, targets)
            shuffled_batch = jax.tree_util.tree_map(
                lambda x: jnp.take(x, permutation, axis=1), batch
            )
            minibatches = jax.tree_util.tree_map(
                lambda x: jnp.swapaxes(
                    jnp.reshape(
                        x,
                        [x.shape[0], config["NUM_MINIBATCHES"], -1] + list(x.shape[2:]),
                    ),
                    1,
                    0,
                ),
                shuffled_batch,
            )

            # UPDATE MINIBATCHES
            (params, opt_state), total_loss = jax.lax.scan(
                _update_minibatch, (params, opt_state), minibatches
            )

            update_state = (
                params,
                opt_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, total_loss

        init_hstate = initial_hstate[None, :]
        update_state = (
            params,
            opt_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )

        # UPDATE EPOCHS
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, config["PPO_EPOCHS"]
        )

        params, opt_state, _, traj_batch, advantages, targets, rng = update_state
        runner_state = (
            params,
            opt_state,
            rng,
            env_state,
            last_timestep,
            last_done,
            hstate,
        )
        metric = traj_batch.info
        return runner_state, metric

    def learner_fn(runner_state: RNNRunnerState) -> ExperimentOutput:
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
                - dones (bool): Whether the initial timestep was a terminal state.
                - hstate (chex.Array): The initial hidden state of the RNN.
        """

        batched_update_step = jax.vmap(
            _update_step, in_axes=(0, None), axis_name="batch"
        )

        runner_state, metric = jax.lax.scan(
            batched_update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return learner_fn


def get_evaluator_fn(env: Environment, apply_fn: callable, config: dict) -> callable:
    """Get the evaluator function."""

    def eval_one_episode(params, runner_state) -> Tuple:
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def _env_step(runner_state: Tuple) -> Tuple:
            """Step the environment."""
            (
                rng,
                env_state,
                last_timestep,
                last_done,
                hstate,
                step_count_,
                return_,
            ) = runner_state

            # PRNG keys.
            rng, policy_rng = jax.random.split(rng)

            # Add a batch dimension and env dimension to the observation.
            batched_observation = jax.tree_util.tree_map(
                lambda x: x[np.newaxis, np.newaxis, :], last_timestep.observation
            )
            ac_in = (batched_observation, last_done[np.newaxis, np.newaxis, :])

            # Run the network.
            hstate, pi, _ = apply_fn(params, hstate, ac_in)

            if config["EVALUATION_GREEDY"]:
                action = pi.mode()
            else:
                action = pi.sample(seed=policy_rng)

            # Step environment.
            env_state, timestep = env.step(env_state, action[-1].squeeze(0))

            # Log episode metrics.
            return_ += timestep.reward
            step_count_ += 1
            runner_state = (
                rng,
                env_state,
                timestep,
                jnp.repeat(timestep.last(), config["NUM_AGENTS"]),
                hstate,
                step_count_,
                return_,
            )
            return runner_state

        def is_done(carry: Tuple) -> jnp.bool_:
            """Check if the episode is done."""
            timestep = carry[2]
            return ~timestep.last()

        rng, env_state, timestep, dones, hstate = runner_state
        return_ = jnp.array(0, float)
        step_count_ = jnp.array(0, int)
        # Add batch dimension to hidden state.
        hstate = jnp.expand_dims(hstate, axis=0)

        final_runner = jax.lax.while_loop(
            is_done,
            _env_step,
            (rng, env_state, timestep, dones, hstate, step_count_, return_),
        )

        rng, env_state, timestep, dones, hstate, step_count_, return_ = final_runner
        eval_metrics = {
            "episode_return": return_,
            "episode_length": step_count_,
        }
        return eval_metrics

    def evaluator_fn(
        trained_params: FrozenDict, rng: chex.PRNGKey
    ) -> Dict[str, Dict[str, chex.Array]]:
        """Evaluator function."""

        # Initialise environment states and timesteps.
        n_devices = len(jax.devices())
        rng, *env_rngs = jax.random.split(
            rng, config["NUM_EVAL_EPISODES"] // n_devices + 1
        )
        env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
            jnp.stack(env_rngs),
        )
        # Split rngs for each core.
        rng, *step_rngs = jax.random.split(
            rng, config["NUM_EVAL_EPISODES"] // n_devices + 1
        )
        # Add dimension to pmap over.
        reshape_step_rngs = lambda x: x.reshape(
            config["NUM_EVAL_EPISODES"] // n_devices, -1
        )
        step_rngs = reshape_step_rngs(jnp.stack(step_rngs))

        # Initialise hidden state.
        init_hstate = ScannedRNN.initialize_carry(
            config["NUM_EVAL_EPISODES"] // n_devices, 128
        )
        init_hstate = jnp.expand_dims(init_hstate, axis=1)
        init_hstate = jnp.tile(init_hstate, (1, config["NUM_AGENTS"], 1))

        # Initialise dones.
        dones = jnp.zeros(
            (
                config["NUM_EVAL_EPISODES"] // n_devices,
                config["NUM_AGENTS"],
            ),
            dtype=bool,
        )

        runner_state = (step_rngs, env_states, timesteps, dones, init_hstate)

        eval_metrics = jax.vmap(
            eval_one_episode, in_axes=(None, 0), axis_name="eval_batch"
        )(trained_params, runner_state)

        return {"metrics": eval_metrics}

    return evaluator_fn


def learner_setup(
    env: Environment, rngs: chex.Array, config: Dict
) -> Tuple[callable, RNNRunnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())
    # Get number of actions and agents.
    num_actions = int(env.action_spec().num_values[0])
    num_agents = env.action_spec().shape[0]
    config["NUM_AGENTS"] = num_agents
    # PRNG keys.
    rng, rng_p = rngs

    # Define network and optimiser.
    network = ActorCritic(num_actions, config["ACTIVATION"])
    optim = optax.adam(config["LR"])

    # Initialise observation: Select only obs for a single agent.
    init_obs = env.observation_spec().generate_value()
    init_obs = jax.tree_util.tree_map(lambda x: x[0], init_obs)
    init_obs = jax.tree_util.tree_map(
        lambda x: jnp.repeat(x[jnp.newaxis, ...], config["NUM_ENVS"], axis=0),
        init_obs,
    )
    init_obs = jax.tree_util.tree_map(lambda x: x[None, ...], init_obs)
    init_done = jnp.zeros((1, config["NUM_ENVS"]), dtype=bool)
    init_x = (init_obs, init_done)

    # Initialise hidden state.
    init_hstate = ScannedRNN.initialize_carry((config["NUM_ENVS"]), 128)

    # initialise params and optimiser state.
    params = network.init(rng_p, init_hstate, init_x)
    opt_state = optim.init(params)

    # Vmap network apply function over number of agents.
    vmapped_network_apply_fn = jax.vmap(
        network.apply, in_axes=(None, 1, 2), out_axes=(1, 2, 2)
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

    # Duplicate the hidden state for each agent.
    init_hstate = jnp.expand_dims(init_hstate, axis=1)
    init_hstate = jnp.tile(init_hstate, (1, config["NUM_AGENTS"], 1))
    hstates = jax.tree_map(broadcast, init_hstate)

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

    # Initialise dones.
    dones = jnp.zeros(
        (
            n_devices,
            config["UPDATE_BATCH_SIZE"],
            config["NUM_ENVS"],
            config["NUM_AGENTS"],
        ),
        dtype=bool,
    )

    return learn, (params, opt_state, step_rngs, env_states, timesteps, dones, hstates)


def evaluator_setup(
    eval_env: Environment, rng_e: chex.PRNGKey, params: FrozenDict, config: Dict
) -> Tuple[callable, RNNRunnerState]:
    """Initialise evaluator_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    n_devices = len(jax.devices())
    # Get number of actions.
    num_actions = int(eval_env.action_spec().num_values[0])

    # Define network and vmap it over number of agents.
    eval_network = ActorCritic(num_actions, config["ACTIVATION"])
    vmapped_eval_network_apply_fn = jax.vmap(
        eval_network.apply, in_axes=(None, 1, 2), out_axes=(1, 2, 2)
    )

    # Pmap evaluator over cores.
    evaluator = get_evaluator_fn(eval_env, vmapped_eval_network_apply_fn, config)
    evaluator = jax.pmap(evaluator, axis_name="device")

    # Broadcast trained params to cores and split rngs for each core.
    trained_params = jax.tree_util.tree_map(lambda x: x[:, 0, ...], params)
    rng_e, *eval_rngs = jax.random.split(rng_e, n_devices + 1)
    eval_rngs = jnp.stack(eval_rngs).reshape(n_devices, -1)

    return evaluator, (trained_params, eval_rngs)


def log(
    logger: SacredLogger,
    metrics_info: Dict[str, Dict[str, chex.Array]],
    episode_count: int = 0,
    t_env: int = 0,
) -> int:
    """Log the episode returns and lengths.

    Args:
        logger (Logger): The logger.
        metrics_info (Dict): The metrics info.
        episode_count (int): The current episode count.
        t_env (int): The current timestep.
    """
    # Flatten metrics info.
    episodes_return = jnp.ravel(metrics_info["episode_return"])
    episodes_length = jnp.ravel(metrics_info["episode_length"])
    # Log metrics.
    print("MEAN EPISODE RETURN: ", np.mean(episodes_return))
    for ep_i in range(episode_count, episode_count + len(episodes_return)):
        logger.log_stat("test_episode_returns", float(episodes_return[ep_i]), ep_i)
        logger.log_stat("test_episode_lengths", float(episodes_length[ep_i]), ep_i)
    episode_count += len(episodes_return)
    logger.log_stat("mean_test_episode_returns", float(np.mean(episodes_return)), t_env)
    logger.log_stat("mean_test_episode_length", float(np.mean(episodes_length)), t_env)
    return episode_count


# Logger setup
logger = get_logger()
ex = Experiment("mava", save_git_info=False)
ex.logger = logger
ex.captured_out_filter = apply_backspaces_and_linefeeds
results_path = os.path.join(dirname(dirname(abspath(__file__))), "results")


@ex.config
def make_config() -> None:
    """Config for the experiment."""
    LR = 2.5e-4
    UPDATE_BATCH_SIZE = 4
    ROLLOUT_LENGTH = 128
    NUM_UPDATES = 10
    NUM_ENVS = 32
    PPO_EPOCHS = 4
    NUM_MINIBATCHES = 8
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    CLIP_EPS = 0.2
    ENT_COEF = 0.01
    VF_COEF = 0.5
    MAX_GRAD_NORM = 0.5
    ACTIVATION = "relu"
    ENV_NAME = "RobotWarehouse-v0"
    SEED = 42
    NUM_EVAL_EPISODES = 32
    NUM_EVALUATION = 5
    EVALUATION_GREEDY = False
    USE_SACRED = True
    USE_TF = True


@ex.main
def run_experiment(_run: Run, _config: Dict, _log: SacredLogger) -> None:
    """Runs experiment."""
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

    # Create envs
    env = jumanji.make(config["ENV_NAME"])
    env = RwareMultiAgentWrapper(env)
    env = AutoResetWrapper(env)
    env = LogWrapper(env)
    eval_env = jumanji.make(config["ENV_NAME"])
    eval_env = RwareMultiAgentWrapper(eval_env)

    # PRNG keys.
    rng, rng_e, rng_p = jax.random.split(jax.random.PRNGKey(config["SEED"]), num=3)
    # Setup learner.
    learn, runner_state = learner_setup(env, (rng, rng_p), config)

    # Setup evaluator.
    evaluator, (trained_params, eval_rngs) = evaluator_setup(
        eval_env, rng_e, runner_state[0], config
    )

    # Calculate total timesteps.
    n_devices = len(jax.devices())
    config["NUM_UPDATES_PER_EVAL"] = config["NUM_UPDATES"] // config["NUM_EVALUATION"]
    timesteps_per_training = (
        n_devices
        * config["NUM_UPDATES_PER_EVAL"]
        * config["ROLLOUT_LENGTH"]
        * config["UPDATE_BATCH_SIZE"]
        * config["NUM_ENVS"]
    )

    # Compile learner and evaluator.
    with TimeIt(tag="COMPILATION"):
        learn(runner_state)
    with TimeIt(tag="COMPILATION"):
        evaluator(trained_params, eval_rngs)

    # Run experiment for a total number of evaluations.
    episode_count = 0
    for i in range(config["NUM_EVALUATION"]):
        # Train.
        with TimeIt(tag="EXECUTION", environment_steps=timesteps_per_training):
            learner_output = learn(runner_state)
            jax.block_until_ready(learner_output)

        # Prepare for evaluation.
        trained_params = jax.tree_util.tree_map(
            lambda x: x[:, 0, ...], learner_output["runner_state"][0]
        )
        rng_e, *eval_rngs = jax.random.split(rng_e, n_devices + 1)
        eval_rngs = jnp.stack(eval_rngs)
        eval_rngs = eval_rngs.reshape(n_devices, -1)

        # Evaluator hidden state.
        init_hstate = ScannedRNN.initialize_carry((config["NUM_ENVS"]), 128)

        # Evaluate.
        evaluator_output = evaluator(trained_params, eval_rngs)
        jax.block_until_ready(evaluator_output)

        # Log the results
        episode_count = log(
            logger=logger,
            metrics_info=evaluator_output["metrics"],
            episode_count=episode_count,
            t_env=timesteps_per_training * (i + 1),
        )

        # Update runner state to continue training.
        runner_state = learner_output["runner_state"]


if __name__ == "__main__":
    file_obs_path = os.path.join(results_path, f"sacred/")
    ex.observers.append(FileStorageObserver.create(file_obs_path))
    ex.run()
    print("Recurrent IPPO experiment completed")
