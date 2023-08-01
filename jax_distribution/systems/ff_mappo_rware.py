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

from jax_distribution.types import ExperimentOutput, PPOTransition, RunnerState
from jax_distribution.utils.jax import merge_leading_dims
from jax_distribution.utils.logger_tools import Logger, config_copy, get_logger
from jax_distribution.utils.timing_utils import TimeIt
from jax_distribution.wrappers.jumanji import LogWrapper, RwareMultiAgentWrapper


class ActorCritic(nn.Module):
    """Actor Critic Network."""

    action_dim: Sequence[int]
    activation: str = "tanh"

    @nn.compact
    def __call__(
        self, gloabal_observation: Observation, observation: Observation
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

        y = gloabal_observation.agents_view
        y = y.reshape(y.shape[0], -1)

        critic_output = nn.Dense(
            64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(y)
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
            actor_policy, value = apply_fn(
                params, last_timestep.observation, last_timestep.observation
            )
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
        _, last_val = apply_fn(
            params, last_timestep.observation, last_timestep.observation
        )

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
                    actor_policy, value = apply_fn(
                        params, traj_batch.obs, traj_batch.obs
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
            batched_update_step, runner_state, None, config["NUM_UPDATES_PER_EVAL"]
        )
        return {"runner_state": runner_state, "metrics": metric}

    return learner_fn


def get_evaluator_fn(env: Environment, apply_fn: callable, config: dict) -> callable:
    """Get the evaluator function."""

    def eval_one_episode(params, runner_state) -> Tuple:
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def _env_step(runner_state: Tuple) -> Tuple:
            """Step the environment."""
            # PRNG keys.
            rng, env_state, last_timestep, step_count_, return_ = runner_state

            # Select action.
            rng, _rng = jax.random.split(rng)
            critic_obs = jax.tree_map(
                lambda x: jnp.expand_dims(x, axis=0), last_timestep.observation
            )
            pi, _ = apply_fn(params, critic_obs, last_timestep.observation)

            if config["EVALUATION_GREEDY"]:
                action = pi.mode()
            else:
                action = pi.sample(seed=_rng)

            # Step environment.
            env_state, timestep = env.step(env_state, action)

            # Log episode metrics.
            return_ += timestep.reward
            step_count_ += 1
            runner_state = (rng, env_state, timestep, step_count_, return_)
            return runner_state

        def is_done(carry: Tuple) -> jnp.bool_:
            """Check if the episode is done."""
            timestep = carry[2]
            return ~timestep.last()

        rng, env_state, timestep = runner_state
        return_ = jnp.array(0, float)
        step_count_ = jnp.array(0, int)

        final_runner = jax.lax.while_loop(
            is_done,
            _env_step,
            (rng, env_state, timestep, step_count_, return_),
        )

        rng, env_state, timestep, step_count_, return_ = final_runner
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
        cores_count = len(jax.devices())
        rng, *env_rngs = jax.random.split(
            rng, config["NUM_EVAL_EPISODES"] // cores_count + 1
        )
        env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
            jnp.stack(env_rngs),
        )
        # Split rngs for each core.
        rng, *step_rngs = jax.random.split(
            rng, config["NUM_EVAL_EPISODES"] // cores_count + 1
        )
        # Add dimension to pmap over.
        reshape_step_rngs = lambda x: x.reshape(
            config["NUM_EVAL_EPISODES"] // cores_count, -1
        )
        step_rngs = reshape_step_rngs(jnp.stack(step_rngs))
        runner_state = (step_rngs, env_states, timesteps)

        eval_metrics = jax.vmap(
            eval_one_episode, in_axes=(None, 0), axis_name="eval_batch"
        )(trained_params, runner_state)

        return {"metrics": eval_metrics}

    return evaluator_fn


def learner_setup(
    env: Environment, rngs: chex.Array, config: Dict
) -> Tuple[callable, RunnerState]:
    """Initialise learner_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    cores_count = len(jax.devices())
    # Get number of actions and agents.
    num_actions = int(env.action_spec().num_values[0])
    num_agents = env.action_spec().shape[0]
    config["NUM_AGENTS"] = num_agents
    # PRNG keys.
    rng, rng_p = rngs

    # Define network and optimiser.
    network = ActorCritic(num_actions, config["ACTIVATION"])
    optim = optax.adam(config["LR"])

    # Initialise observation.
    init_x = env.observation_spec().generate_value()
    init_y = init_x
    # Select only obs for a single agent.
    init_x = jax.tree_util.tree_map(lambda x: x[0], init_x)
    init_x = jax.tree_util.tree_map(lambda x: x[None, ...], init_x)
    # Select obs for all the agents.
    init_y = jax.tree_util.tree_map(lambda y: y[None, ...], init_y)

    # initialise params and optimiser state.
    params = network.init(rng_p, init_y, init_x)
    opt_state = optim.init(params)

    # Vmap network apply function over number of agents.
    vmapped_network_apply_fn = jax.vmap(
        network.apply, in_axes=(None, None, 1), out_axes=(1, 1)
    )

    # Get batched iterated update and replicate it to pmap it over cores.
    learn = get_learner_fn(env, vmapped_network_apply_fn, optim.update, config)
    learn = jax.pmap(learn, axis_name="device")

    # Broadcast params and optimiser state to cores and batch.
    broadcast = lambda x: jnp.broadcast_to(
        x, (cores_count, config["UPDATE_BATCH_SIZE"]) + x.shape
    )
    params = jax.tree_map(broadcast, params)
    opt_state = jax.tree_map(broadcast, opt_state)

    # Initialise environment states and timesteps.
    rng, *env_rngs = jax.random.split(
        rng, cores_count * config["UPDATE_BATCH_SIZE"] * config["NUM_ENVS"] + 1
    )
    env_states, timesteps = jax.vmap(env.reset, in_axes=(0))(
        jnp.stack(env_rngs),
    )

    # Split rngs for each core.
    rng, *step_rngs = jax.random.split(
        rng, cores_count * config["UPDATE_BATCH_SIZE"] + 1
    )
    # Add dimension to pmap over.
    reshape_step_rngs = lambda x: x.reshape(
        (cores_count, config["UPDATE_BATCH_SIZE"]) + x.shape[1:]
    )
    step_rngs = reshape_step_rngs(jnp.stack(step_rngs))
    reshape_states = lambda x: x.reshape(
        (cores_count, config["UPDATE_BATCH_SIZE"], config["NUM_ENVS"]) + x.shape[1:]
    )
    env_states = jax.tree_util.tree_map(reshape_states, env_states)
    timesteps = jax.tree_util.tree_map(reshape_states, timesteps)

    return learn, (params, opt_state, step_rngs, env_states, timesteps)


def evaluator_setup(
    eval_env: Environment, rng_e: chex.PRNGKey, params: FrozenDict, config: Dict
) -> Tuple[callable, RunnerState]:
    """Initialise evaluator_fn, network, optimiser, environment and states."""
    # Get available TPU cores.
    cores_count = len(jax.devices())
    # Get number of actions.
    num_actions = int(eval_env.action_spec().num_values[0])

    # Define network and vmap it over number of agents.
    eval_network = ActorCritic(num_actions, config["ACTIVATION"])

    # Vmap network apply function over number of agents.
    vmapped_eval_network_apply_fn = jax.vmap(
        eval_network.apply, in_axes=(None, None, 0)
    )

    # Pmap evaluator over cores.
    evaluator = get_evaluator_fn(eval_env, vmapped_eval_network_apply_fn, config)
    evaluator = jax.pmap(evaluator, axis_name="device")

    # Broadcast trained params to cores and split rngs for each core.
    trained_params = jax.tree_util.tree_map(lambda x: x[:, 0, ...], params)
    rng_e, *eval_rngs = jax.random.split(rng_e, cores_count + 1)
    eval_rngs = jnp.stack(eval_rngs).reshape(cores_count, -1)

    return evaluator, (trained_params, eval_rngs)


def log(
    logger: Logger,
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
    NUM_UPDATES = 20
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
    cores_count = len(jax.devices())
    config["NUM_UPDATES_PER_EVAL"] = config["NUM_UPDATES"] // config["NUM_EVALUATION"]
    timesteps_per_training = (
        cores_count
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
        rng_e, *eval_rngs = jax.random.split(rng_e, cores_count + 1)
        eval_rngs = jnp.stack(eval_rngs)
        eval_rngs = eval_rngs.reshape(cores_count, -1)

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
    print("MAPPO experiment completed")
