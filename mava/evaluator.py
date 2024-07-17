# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

from typing import Any, Callable, Dict, Optional, Tuple, Union

import chex
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from omegaconf import DictConfig

from mava.types import (
    ActorApply,
    EvalFn,
    EvalState,
    ExperimentOutput,
    Observation,
    RecActorApply,
    RNNEvalState,
    RNNObservation,
    SebulbaEvalFn,
)


def get_anakin_ff_evaluator_fn(
    env: Environment,
    apply_fn: ActorApply,
    config: DictConfig,
    log_win_rate: bool = False,
    eval_multiplier: int = 1,
) -> EvalFn:
    """Get the evaluator function for feedforward networks.

    Args:
        env (Environment): An evironment instance for evaluation.
        apply_fn (callable): Network forward pass method.
        config (dict): Experiment configuration.
        eval_multiplier (int): A scalar that will increase the number of evaluation
            episodes by a fixed factor. The reason for the increase is to enable the
            computation of the `absolute metric` which is a metric computed and the end
            of training by rolling out the policy which obtained the greatest evaluation
            performance during training for 10 times more episodes than were used at a
            single evaluation step.
    """

    def eval_one_episode(params: FrozenDict, init_eval_state: EvalState) -> Dict:
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def _env_step(eval_state: EvalState) -> EvalState:
            """Step the environment."""
            # PRNG keys.
            key, env_state, last_timestep, step_count, episode_return = eval_state

            # Select action.
            key, policy_key = jax.random.split(key)
            # Add a batch dimension to the observation.
            pi = apply_fn(
                params, jax.tree_map(lambda x: x[jnp.newaxis, ...], last_timestep.observation)
            )

            if config.arch.evaluation_greedy:
                action = pi.mode()
            else:
                action = pi.sample(seed=policy_key)

            # Remove batch dim for stepping the environment.
            action = jnp.squeeze(action, axis=0)

            # Step environment.
            env_state, timestep = env.step(env_state, action)

            # Log episode metrics.
            episode_return += timestep.reward
            step_count += 1
            eval_state = EvalState(key, env_state, timestep, step_count, episode_return)
            return eval_state

        def not_done(carry: Tuple) -> bool:
            """Check if the episode is done."""
            timestep = carry[2]
            is_not_done: bool = ~timestep.last()
            return is_not_done

        final_state = jax.lax.while_loop(not_done, _env_step, init_eval_state)

        eval_metrics = {
            "episode_return": final_state.episode_return,
            "episode_length": final_state.step_count,
        }
        # Log won episode if win rate is required.

        if log_win_rate:
            eval_metrics["won_episode"] = final_state.timestep.extras["won_episode"]

        return eval_metrics

    def evaluator_fn(trained_params: FrozenDict, key: chex.PRNGKey) -> ExperimentOutput[EvalState]:
        """Evaluator function."""

        # Initialise environment states and timesteps.
        n_devices = len(jax.devices())

        eval_batch = (config.arch.num_eval_episodes // n_devices) * eval_multiplier

        key, *env_keys = jax.random.split(key, eval_batch + 1)
        env_states, timesteps = jax.vmap(env.reset)(jnp.stack(env_keys))
        # Split keys for each core.
        key, *step_keys = jax.random.split(key, eval_batch + 1)
        # Add dimension to pmap over.
        step_keys = jnp.stack(step_keys).reshape(eval_batch, -1)

        eval_state = EvalState(
            key=step_keys,
            env_state=env_states,
            timestep=timesteps,
            step_count=jnp.zeros((eval_batch, 1)),
            episode_return=jnp.zeros_like(timesteps.reward),
        )

        eval_metrics = jax.vmap(
            eval_one_episode,
            in_axes=(None, 0),
            axis_name="eval_batch",
        )(trained_params, eval_state)

        return ExperimentOutput(
            learner_state=eval_state,
            episode_metrics=eval_metrics,
            train_metrics={},
        )

    return evaluator_fn


def get_anakin_rnn_evaluator_fn(
    env: Environment,
    apply_fn: RecActorApply,
    config: DictConfig,
    scanned_rnn: nn.Module,
    log_win_rate: bool = False,
    eval_multiplier: int = 1,
) -> EvalFn:
    """Get the evaluator function for recurrent networks."""

    def eval_one_episode(params: FrozenDict, init_eval_state: RNNEvalState) -> Dict:
        """Evaluate one episode. It is vectorized over the number of evaluation episodes."""

        def _env_step(eval_state: RNNEvalState) -> RNNEvalState:
            """Step the environment."""
            (
                key,
                env_state,
                last_timestep,
                last_done,
                hstate,
                step_count,
                episode_return,
            ) = eval_state

            # PRNG keys.
            key, policy_key = jax.random.split(key)

            # Add a batch dimension and env dimension to the observation.
            batched_observation = jax.tree_util.tree_map(
                lambda x: x[jnp.newaxis, jnp.newaxis, :], last_timestep.observation
            )
            ac_in = (
                batched_observation,
                last_done[jnp.newaxis, jnp.newaxis, :],
            )

            # Run the network.
            hstate, pi = apply_fn(params, hstate, ac_in)

            if config.arch.evaluation_greedy:
                action = pi.mode()
            else:
                action = pi.sample(seed=policy_key)

            # Step environment.
            env_state, timestep = env.step(env_state, action[-1].squeeze(0))

            # Log episode metrics.
            episode_return += timestep.reward
            step_count += 1
            eval_state = RNNEvalState(
                key,
                env_state,
                timestep,
                jnp.repeat(timestep.last(), config.system.num_agents),
                hstate,
                step_count,
                episode_return,
            )
            return eval_state

        def not_done(carry: Tuple) -> bool:
            """Check if the episode is done."""
            timestep = carry[2]
            is_not_done: bool = ~timestep.last()
            return is_not_done

        final_state = jax.lax.while_loop(not_done, _env_step, init_eval_state)

        eval_metrics = {
            "episode_return": final_state.episode_return,
            "episode_length": final_state.step_count,
        }
        # Log won episode if win rate is required.
        if log_win_rate:
            eval_metrics["won_episode"] = final_state.timestep.extras["won_episode"]

        return eval_metrics

    def evaluator_fn(
        trained_params: FrozenDict, key: chex.PRNGKey
    ) -> ExperimentOutput[RNNEvalState]:
        """Evaluator function."""

        # Initialise environment states and timesteps.
        n_devices = len(jax.devices())

        eval_batch = config.arch.num_eval_episodes // n_devices * eval_multiplier

        key, *env_keys = jax.random.split(key, eval_batch + 1)
        env_states, timesteps = jax.vmap(env.reset)(jnp.stack(env_keys))
        # Split keys for each core.
        key, *step_keys = jax.random.split(key, eval_batch + 1)
        # Add dimension to pmap over.
        step_keys = jnp.stack(step_keys).reshape(eval_batch, -1)

        # Initialise hidden state.
        init_hstate = scanned_rnn.initialize_carry(
            (eval_batch, config.system.num_agents),
            config.network.hidden_state_dim,
        )

        # Initialise dones.
        dones = jnp.zeros(
            (
                eval_batch,
                config.system.num_agents,
            ),
            dtype=bool,
        )

        # Adding an extra batch dim for the vmapped eval functions.
        init_hstate = init_hstate[:, jnp.newaxis, ...]

        eval_state = RNNEvalState(
            key=step_keys,
            env_state=env_states,
            timestep=timesteps,
            dones=dones,
            hstate=init_hstate,
            step_count=jnp.zeros((eval_batch, 1)),
            episode_return=jnp.zeros_like(timesteps.reward),
        )

        eval_metrics = jax.vmap(
            eval_one_episode,
            in_axes=(None, 0),
            axis_name="eval_batch",
        )(trained_params, eval_state)

        return ExperimentOutput(
            learner_state=eval_state,
            episode_metrics=eval_metrics,
            train_metrics={},
        )

    return evaluator_fn


def make_anakin_eval_fns(
    eval_env: Environment,
    network_apply_fn: Union[ActorApply, RecActorApply],
    config: DictConfig,
    use_recurrent_net: bool = False,
    scanned_rnn: Optional[nn.Module] = None,
) -> Tuple[EvalFn, EvalFn]:
    """Initialize evaluator functions for reinforcement learning.

    Args:
        eval_env (Environment): The environment used for evaluation.
        network_apply_fn (Union[ActorApply,RecActorApply]): Creates a policy to sample.
        config (DictConfig): The configuration settings for the evaluation.
        use_recurrent_net (bool, optional): Whether to use a rnn. Defaults to False.
        scanned_rnn (Optional[nn.Module], optional): The rnn module.
            Required if `use_recurrent_net` is True. Defaults to None.

    Returns:
        Tuple[EvalFn, EvalFn]: A tuple of two evaluation functions:
        one for use during training and one for absolute metrics.

    Raises:
        AssertionError: If `use_recurrent_net` is True but `scanned_rnn` is not provided.
    """
    # Check if win rate is required for evaluation.
    log_win_rate = config.env.log_win_rate
    # Vmap it over number of agents and create evaluator_fn.
    if use_recurrent_net:
        assert scanned_rnn is not None
        evaluator = get_anakin_rnn_evaluator_fn(
            eval_env,
            network_apply_fn,  # type: ignore
            config,
            scanned_rnn,
            log_win_rate,
        )
        absolute_metric_evaluator = get_anakin_rnn_evaluator_fn(
            eval_env,
            network_apply_fn,  # type: ignore
            config,
            scanned_rnn,
            log_win_rate,
            10,
        )
    else:
        evaluator = get_anakin_ff_evaluator_fn(
            eval_env, network_apply_fn, config, log_win_rate  # type: ignore
        )
        absolute_metric_evaluator = get_anakin_ff_evaluator_fn(
            eval_env, network_apply_fn, config, log_win_rate, 10  # type: ignore
        )

    evaluator = jax.pmap(evaluator, axis_name="device")
    absolute_metric_evaluator = jax.pmap(absolute_metric_evaluator, axis_name="device")

    return evaluator, absolute_metric_evaluator


def get_sebulba_ff_evaluator_fn(
    env: Environment,
    apply_fn: ActorApply,
    config: DictConfig,
    np_rng: np.random.Generator,
    log_win_rate: bool = False,
) -> SebulbaEvalFn:
    """Get the evaluator function for feedforward networks.

    Args:
        env (Environment): An evironment instance for evaluation.
        apply_fn (callable): Network forward pass method.
        config (dict): Experiment configuration.
    """

    @jax.jit
    def get_action(  # todo explicetly put these on the learner? they should already be there
        params: FrozenDict,
        observation: Observation,
        key: chex.PRNGKey,
    ) -> chex.Array:
        """Get action."""

        pi = apply_fn(params, observation)

        if config.arch.evaluation_greedy:
            action = pi.mode()
        else:
            action = pi.sample(seed=key)

        return action

    def eval_episodes(params: FrozenDict, key: chex.PRNGKey) -> Any:

        seeds = np_rng.integers(np.iinfo(np.int64).max, size=env.num_envs).tolist()
        obs, info = env.reset(seed=seeds)
        dones = np.full(env.num_envs, False)
        eval_metrics = jax.tree_map(lambda *x: jnp.asarray(x), *info["metrics"])

        while not dones.all():

            key, policy_key = jax.random.split(key)

            obs = jax.device_put(jnp.stack(obs, axis=1))
            action_mask = jax.device_put(np.stack(info["actions_mask"]))

            actions = get_action(params, Observation(obs, action_mask), policy_key)
            cpu_action = jax.device_get(actions)

            obs, reward, terminated, truncated, info = env.step(cpu_action.swapaxes(0, 1))

            next_metrics = jax.tree_map(lambda *x: jnp.asarray(x), *info["metrics"])

            next_dones = next_metrics["is_terminal_step"]

            update_flags = np.logical_and(next_dones, np.invert(dones))

            update_metrics = lambda new_metric, old_metric, update_flags=update_flags: np.where(
                (update_flags), new_metric, old_metric
            )

            eval_metrics = jax.tree_map(update_metrics, next_metrics, eval_metrics)

            dones = np.logical_or(dones, next_dones)
        eval_metrics.pop("is_terminal_step")

        return eval_metrics

    return eval_episodes


def get_sebulba_rnn_evaluator_fn(
    env: Environment,
    apply_fn: RecActorApply,
    config: DictConfig,
    np_rng: np.random.Generator,
    scanned_rnn: nn.Module,
    log_win_rate: bool = False,
) -> SebulbaEvalFn:
    """Get the evaluator function for feedforward networks.

    Args:
        env (Environment): An evironment instance for evaluation.
        apply_fn (callable): Network forward pass method.
        config (dict): Experiment configuration.
    """

    @jax.jit
    def get_action(  # todo explicetly put these on the learner? they should already be there
        params: FrozenDict,
        observation: RNNObservation,
        hstate: chex.Array,
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array]:
        """Get action."""

        hstate, pi = apply_fn(params, hstate, observation)

        if config.arch.evaluation_greedy:
            action = pi.mode()
        else:
            action = pi.sample(seed=key)

        return action, hstate

    def eval_episodes(params: FrozenDict, key: chex.PRNGKey) -> Any:

        seeds = np_rng.integers(np.iinfo(np.int64).max, size=env.num_envs).tolist()
        obs, info = env.reset(seed=seeds)
        eval_metrics = jax.tree_map(lambda *x: jnp.asarray(x), *info["metrics"])

        hstate = scanned_rnn.initialize_carry(
            (env.num_envs, config.system.num_agents), config.network.hidden_state_dim
        )

        dones = jnp.full((env.num_envs, config.system.num_agents), False)

        while not dones.all():

            key, policy_key = jax.random.split(key)

            obs = jax.device_put(jnp.stack(obs, axis=1))
            action_mask = jax.device_put(np.stack(info["actions_mask"]))

            obs, action_mask, dones = jax.tree_map(
                lambda x: x[jnp.newaxis, :], (obs, action_mask, dones)
            )

            actions, hstate = get_action(
                params, (Observation(obs, action_mask), dones), hstate, policy_key
            )
            cpu_action = jax.device_get(actions)

            obs, reward, terminated, truncated, info = env.step(cpu_action[0].swapaxes(0, 1))

            next_metrics = jax.tree_map(lambda *x: jnp.asarray(x), *info["metrics"])

            next_dones = np.logical_or(terminated, truncated)

            update_flags = np.all(np.logical_and(next_dones, np.invert(dones[0])), axis=1)

            update_metrics = lambda new_metric, old_metric, update_flags=update_flags: np.where(
                (update_flags), new_metric, old_metric
            )

            eval_metrics = jax.tree_map(update_metrics, next_metrics, eval_metrics)

            dones = np.logical_or(dones, next_dones)
        eval_metrics.pop("is_terminal_step")

        return eval_metrics

    return eval_episodes


def make_sebulba_eval_fns(
    eval_env_fn: Callable,
    network_apply_fn: Union[ActorApply, RecActorApply],
    config: DictConfig,
    np_rng: np.random.Generator,
    add_global_state: bool = False,
    use_recurrent_net: bool = False,
    scanned_rnn: Optional[nn.Module] = None,
) -> Tuple[SebulbaEvalFn, SebulbaEvalFn]:
    """Initialize evaluator functions for reinforcement learning.

    Args:
        eval_env_fn (Environment): The function to Create the eval envs.
        network_apply_fn (Union[ActorApply,RecActorApply]): Creates a policy to sample.
        config (DictConfig): The configuration settings for the evaluation.
        use_recurrent_net (bool, optional): Whether to use a rnn. Defaults to False.
        scanned_rnn (Optional[nn.Module], optional): The rnn module.
            Required if `use_recurrent_net` is True. Defaults to None.

    Returns:
        Tuple[SebulbaEvalFn, SebulbaEvalFn]: A tuple of two evaluation functions:
        one for use during training and one for absolute metrics.

    Raises:
        AssertionError: If `use_recurrent_net` is True but `scanned_rnn` is not provided.
    """
    eval_env, absolute_eval_env = eval_env_fn(
        config, config.arch.num_eval_episodes, add_global_state=add_global_state
    ), eval_env_fn(config, config.arch.num_eval_episodes * 10, add_global_state=add_global_state)

    # Check if win rate is required for evaluation.
    log_win_rate = config.env.log_win_rate
    # Vmap it over number of agents and create evaluator_fn.
    if use_recurrent_net:
        assert scanned_rnn is not None
        evaluator = get_sebulba_rnn_evaluator_fn(
            eval_env,
            network_apply_fn,  # type: ignore
            config,
            np_rng,
            scanned_rnn,
            log_win_rate,
        )
        absolute_metric_evaluator = get_sebulba_rnn_evaluator_fn(
            absolute_eval_env,
            network_apply_fn,  # type: ignore
            config,
            np_rng,
            scanned_rnn,
            log_win_rate,
        )
    else:
        evaluator = get_sebulba_ff_evaluator_fn(
            eval_env, network_apply_fn, config, np_rng, log_win_rate  # type: ignore
        )
        absolute_metric_evaluator = get_sebulba_ff_evaluator_fn(
            absolute_eval_env, network_apply_fn, config, np_rng, log_win_rate  # type: ignore
        )

    return evaluator, absolute_metric_evaluator
