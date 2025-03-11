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

import math
import time
import warnings
from typing import Any, Callable, Dict, Protocol, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from jax import tree
from jumanji.types import TimeStep
from omegaconf import DictConfig
from typing_extensions import TypeAlias

from mava.types import (
    Action,
    ActorApply,
    MarlEnv,
    Metrics,
    Observation,
    ObservationGlobalState,
    RecActorApply,
    State,
)
from mava.wrappers.gym import GymToJumanji

# Optional extras that are passed out of the actor and then into the actor in the next step
ActorState: TypeAlias = Dict[str, Any]
# Type of the carry for the _env_step function in the evaluator
_EvalEnvStepState: TypeAlias = Tuple[State, TimeStep, PRNGKey, ActorState]
# The function signature for the mava evaluation function (returned by `get_eval_fn`).
EvalFn: TypeAlias = Callable[[FrozenDict, PRNGKey, ActorState], Metrics]


class EvalActFn(Protocol):
    """The API for the acting function that is passed to the `EvalFn`.

    A get_action function must conform to this API in order to be used with Mava's evaluator.
    See `make_ff_eval_act_fn` and `make_rec_eval_act_fn` as examples.
    """

    def __call__(
        self,
        params: FrozenDict,
        timestep: TimeStep[Union[Observation, ObservationGlobalState]],
        key: PRNGKey,
        actor_state: ActorState,
    ) -> Tuple[Array, ActorState]: ...


def get_num_eval_envs(config: DictConfig, absolute_metric: bool) -> int:
    """Returns the number of vmapped envs/batch size during evaluation."""
    n_devices = jax.device_count()
    n_parallel_envs = config.arch.num_envs * n_devices

    if absolute_metric:
        eval_episodes = config.arch.num_absolute_metric_eval_episodes
    else:
        eval_episodes = config.arch.num_eval_episodes

    if eval_episodes <= n_parallel_envs:
        return math.ceil(eval_episodes / n_devices)  # type: ignore
    else:
        return config.arch.num_envs  # type: ignore


def get_eval_fn(
    env: MarlEnv, act_fn: EvalActFn, config: DictConfig, absolute_metric: bool
) -> EvalFn:
    """Creates a function that can be used to evaluate agents on a given environment.

    Args:
    ----
        env: an environment that conforms to the mava environment spec.
        act_fn: a function that takes in params, timestep, key and optionally a state
                and returns actions and optionally a state (see `EvalActFn`).
        config: the system config.
        absolute_metric: whether or not this evaluator calculates the absolute_metric.
                This determines how many evaluation episodes it does.
    """
    n_devices = jax.device_count()
    eval_episodes = (
        config.arch.num_absolute_metric_eval_episodes
        if absolute_metric
        else config.arch.num_eval_episodes
    )
    n_vmapped_envs = get_num_eval_envs(config, absolute_metric)
    n_parallel_envs = n_vmapped_envs * n_devices
    episode_loops = math.ceil(eval_episodes / n_parallel_envs)

    # Warnings if num eval episodes is not divisible by num parallel envs.
    if eval_episodes % n_parallel_envs != 0:
        warnings.warn(
            f"Number of evaluation episodes ({eval_episodes}) is not divisible by `num_envs` * "
            f"`num_devices` ({n_parallel_envs} * {n_devices}). Some extra evaluations will be "
            f"executed. New number of evaluation episodes = {episode_loops * n_parallel_envs}",
            stacklevel=2,
        )

    def eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Evaluates the given params on an environment and returns relevent metrics.

        Metrics are collected by the `RecordEpisodeMetrics` wrapper: episode return and length,
        also win rate for environments that support it.

        Returns: Dict[str, Array] - dictionary of metric name to metric values for each episode.
        """

        def _env_step(eval_state: _EvalEnvStepState, _: Any) -> Tuple[_EvalEnvStepState, TimeStep]:
            """Performs a single environment step"""
            env_state, ts, key, actor_state = eval_state

            key, act_key = jax.random.split(key)
            action, actor_state = act_fn(params, ts, act_key, actor_state)
            env_state, ts = jax.vmap(env.step)(env_state, action)

            return (env_state, ts, key, actor_state), ts

        def _episode(key: PRNGKey, _: Any) -> Tuple[PRNGKey, Metrics]:
            """Simulates `num_envs` episodes."""
            key, reset_key = jax.random.split(key)
            reset_keys = jax.random.split(reset_key, n_vmapped_envs)
            env_state, ts = jax.vmap(env.reset)(reset_keys)

            step_state = env_state, ts, key, init_act_state
            _, timesteps = jax.lax.scan(_env_step, step_state, jnp.arange(env.time_limit + 1))

            metrics = timesteps.extras["episode_metrics"]
            if config.env.log_win_rate:
                metrics["won_episode"] = timesteps.extras["won_episode"]

            # find the first instance of done to get the metrics at that timestep, we don't
            # care about subsequent steps because we only the results from the first episode
            done_idx = jnp.argmax(timesteps.last(), axis=0)
            metrics = tree.map(lambda m: m[done_idx, jnp.arange(n_vmapped_envs)], metrics)
            del metrics["is_terminal_step"]  # uneeded for logging

            return key, metrics

        # This loop is important because we don't want too many parallel envs.
        # So in evaluation we have num_envs parallel envs and loop enough times
        # so that we do at least `eval_episodes` number of episodes.
        _, metrics = jax.lax.scan(_episode, key, xs=None, length=episode_loops)
        metrics = tree.map(lambda x: x.reshape(-1), metrics)  # flatten metrics
        return metrics

    def timed_eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Wrapper around eval function to time it and add in steps per second metric."""
        start_time = time.time()

        metrics = jax.pmap(eval_fn)(params, key, init_act_state)
        metrics = jax.block_until_ready(metrics)

        end_time = time.time()
        total_timesteps = jnp.sum(metrics["episode_length"])
        metrics["steps_per_second"] = total_timesteps / (end_time - start_time)
        return metrics

    return timed_eval_fn


def make_ff_eval_act_fn(actor_apply_fn: ActorApply, config: DictConfig) -> EvalActFn:
    """Makes an act function that conforms to the evaluator API given a standard
    feed forward mava actor network."""

    def eval_act_fn(
        params: FrozenDict, timestep: TimeStep, key: PRNGKey, actor_state: ActorState
    ) -> Tuple[Action, Dict]:
        pi = actor_apply_fn(params, timestep.observation)
        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
        return action, {}

    return eval_act_fn


def make_rec_eval_act_fn(actor_apply_fn: RecActorApply, config: DictConfig) -> EvalActFn:
    """Makes an act function that conforms to the evaluator API given a standard
    recurrent mava actor network."""

    _hidden_state = "hidden_state"

    def eval_act_fn(
        params: FrozenDict, timestep: TimeStep, key: PRNGKey, actor_state: ActorState
    ) -> Tuple[Action, Dict]:
        hidden_state = actor_state[_hidden_state]

        n_agents = timestep.observation.agents_view.shape[1]
        last_done = timestep.last()[:, jnp.newaxis].repeat(n_agents, axis=-1)
        ac_in = (timestep.observation, last_done)
        ac_in = tree.map(lambda x: x[jnp.newaxis], ac_in)  # add batch dim to obs

        hidden_state, pi = actor_apply_fn(params, hidden_state, ac_in)
        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
        return action.squeeze(0), {_hidden_state: hidden_state}

    return eval_act_fn


def get_sebulba_eval_fn(
    env_maker: Callable[[int, int], GymToJumanji],
    act_fn: EvalActFn,
    config: DictConfig,
    np_rng: np.random.Generator,
    absolute_metric: bool,
) -> Tuple[EvalFn, Any]:
    """Creates a function that can be used to evaluate agents on a given environment.

    Args:
    ----
        env_maker: A function to create the environment instances.
        act_fn: A function that takes in params, timestep, key and optionally a state
                and returns actions and optionally a state (see `EvalActFn`).
        config: The system config.
        np_rng: Random number generator for seeding environment.
        absolute_metric: Whether or not this evaluator calculates the absolute_metric.
                This determines how many evaluation episodes it does.
    """
    n_devices = jax.device_count()
    eval_episodes = (
        config.arch.num_absolute_metric_eval_episodes
        if absolute_metric
        else config.arch.num_eval_episodes
    )

    n_parallel_envs = min(eval_episodes, config.arch.num_envs)
    episode_loops = math.ceil(eval_episodes / n_parallel_envs)
    env = env_maker(config, n_parallel_envs)

    act_fn = jax.jit(
        act_fn, device=jax.local_devices()[config.arch.actor_device_ids[0]]
    )  # Evaluate using the first actor device

    # Warnings if num eval episodes is not divisible by num parallel envs.
    if eval_episodes % n_parallel_envs != 0:
        warnings.warn(
            f"Number of evaluation episodes ({eval_episodes}) is not divisible by `num_envs` * "
            f"`num_devices` ({n_parallel_envs} * {n_devices}). Some extra evaluations will be "
            f"executed. New number of evaluation episodes = {episode_loops * n_parallel_envs}",
            stacklevel=2,
        )

    def eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Evaluates the given params on an environment and returns relevent metrics.

        Metrics are collected by the `RecordEpisodeMetrics` wrapper: episode return and length,
        also win rate for environments that support it.

        Returns: Dict[str, Array] - dictionary of metric name to metric values for each episode.
        """

        def _episode(key: PRNGKey) -> Tuple[PRNGKey, Metrics]:
            """Simulates `num_envs` episodes."""

            # Generate a list of random seeds within the 32-bit integer range, using a seeded RNG.
            seeds = np_rng.integers(np.iinfo(np.int32).max, size=n_parallel_envs).tolist()
            ts = env.reset(seed=seeds)

            timesteps_array = [ts]

            actor_state = init_act_state
            finished_eps = ts.last()

            while not finished_eps.all():
                key, act_key = jax.random.split(key)
                action, actor_state = act_fn(params, ts, act_key, actor_state)
                cpu_action = jax.device_get(action)
                ts = env.step(cpu_action)
                timesteps_array.append(ts)

                finished_eps = np.logical_or(finished_eps, ts.last())

            timesteps = jax.tree.map(lambda *x: np.stack(x), *timesteps_array)

            metrics = timesteps.extras["episode_metrics"]
            if config.env.log_win_rate:
                metrics["won_episode"] = timesteps.extras["won_episode"]

            # Find the first instance of done to get the metrics at that timestep.
            done_idx = np.argmax(timesteps.last(), axis=0)
            metrics = tree.map(lambda m: m[done_idx, np.arange(n_parallel_envs)], metrics)
            del metrics["is_terminal_step"]  # uneeded for logging

            return key, metrics

        # This loop is important because we don't want too many parallel envs.
        # So in evaluation we have num_envs parallel envs and loop enough times
        # so that we do at least `eval_episodes` number of episodes.
        metrics_array = []
        for _ in range(episode_loops):
            key, metric = _episode(key)
            metrics_array.append(metric)

        # flatten metrics
        metrics: Metrics = tree.map(lambda *x: np.array(x).reshape(-1), *metrics_array)
        return metrics

    def timed_eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Metrics:
        """Wrapper around eval function to time it and add in steps per second metric."""
        start_time = time.time()

        metrics = eval_fn(params, key, init_act_state)

        end_time = time.time()
        total_timesteps = jnp.sum(metrics["episode_length"])
        metrics["steps_per_second"] = total_timesteps / (end_time - start_time)
        return metrics

    return timed_eval_fn, env
