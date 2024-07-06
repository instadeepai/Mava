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
import warnings
from typing import Any, Dict, Protocol, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from flax.core.frozen_dict import FrozenDict
from jumanji.env import Environment
from jumanji.types import TimeStep
from omegaconf import DictConfig
from typing_extensions import TypeAlias

from mava.types import Action, Metrics, Observation, ObservationGlobalState, State

# Optional extras that are passed out of the actor and then into the actor in the next step
ActorState: TypeAlias = Dict[str, Any]
# Type of the carry for the _env_step function in the evaluator
_EvalEnvStepState: TypeAlias = Tuple[State, TimeStep, PRNGKey, ActorState]


class EvalActFn(Protocol):
    """The API for the acting function that is passed to the `EvalFn`.

    Your get_action fucntion must conform to this API in order to be used with Mava's evaluator.
    See `make_ff_eval_act_fn` and `make_rec_eval_act_fn` as examples.
    """

    def __call__(  # noqa: E704
        self,
        params: FrozenDict,
        timestep: TimeStep[Union[Observation, ObservationGlobalState]],
        key: PRNGKey,
        **actor_state: Array,
    ) -> Tuple[Array, ActorState]: ...


class EvalFn(Protocol):
    """The function signature for the evaluation function returned by `get_eval_fn`."""

    def __call__(  # noqa: E704
        self, params: FrozenDict, key: PRNGKey, actor_state: ActorState
    ) -> Metrics: ...


def get_eval_fn(
    env: Environment, act_fn: EvalActFn, config: DictConfig, absolute_metric: bool
) -> EvalFn:
    """Creates a function that can be used to evaluate agents on a given environment.

    Args:
        env: an environment that conforms to the mava environment spec.
        act_fn: a function that takes in params, timestep, key and optionally a state
                and returns actions and optionally a state (see `EvalActFn`).
        config: the system config.
        absolute_metric: whether or not this evaluator calculates the absolute_metric.
                This determines how many evaluation episodes it does.
    """
    # Calculating how many eval loops and parallel environments to have
    n_devices = jax.device_count()
    eval_episodes = (
        config.arch.num_abs_metric_eval_episodes
        if absolute_metric
        else config.arch.num_eval_episodes
    )
    if eval_episodes < config.arch.num_envs * n_devices:
        warnings.warn(
            f"Number of evaluation episodes ({eval_episodes}) is less than"
            f"`num_envs` * `num_devices` ({config.arch.num_envs} * {n_devices}). "
            f"Automatically reducing number of envs.",
            stacklevel=2,
        )
        num_envs = math.ceil(eval_episodes / n_devices)
    else:
        num_envs = config.arch.num_envs

    if eval_episodes % (num_envs * n_devices) != 0:
        warnings.warn(
            f"Number of evaluation episodes ({eval_episodes}) is not divisible by "
            f"`num_envs` * `num_devices` ({num_envs} * {n_devices})",
            stacklevel=2,
        )
    episode_loops = math.ceil(eval_episodes / (num_envs * n_devices))

    def eval_fn(params: FrozenDict, key: PRNGKey, init_act_state: ActorState) -> Dict[str, Array]:
        def _env_step(eval_state: _EvalEnvStepState, _: Any) -> Tuple[_EvalEnvStepState, TimeStep]:
            """Performs a single environment step"""
            env_state, ts, key, actor_state = eval_state

            key, act_key = jax.random.split(key)
            action, actor_state = act_fn(params, ts, act_key, **actor_state)
            env_state, ts = jax.vmap(env.step)(env_state, action)

            return (env_state, ts, key, actor_state), ts

        def _episode(key: PRNGKey, _: Any) -> Tuple[PRNGKey, Metrics]:
            """Simulates `num_envs` episodes."""
            key, reset_key = jax.random.split(key)
            reset_keys = jax.random.split(reset_key, num_envs)
            env_state, ts = jax.vmap(env.reset)(reset_keys)

            step_state = env_state, ts, key, init_act_state
            _, timesteps = jax.lax.scan(_env_step, step_state, jnp.arange(env.time_limit))

            metrics = timesteps.extras["episode_metrics"]
            if config.env.log_win_rate:
                metrics["won_episode"] = timesteps.extras["won_episode"]

            # find the first instance of done to get the metrics at that timestep we don't
            # care about subsequent steps because we only the results from the first episode
            done_idx = jnp.argmax(timesteps.last(), axis=0)
            metrics = jax.tree_map(lambda m: m[done_idx, jnp.arange(num_envs)], metrics)
            del metrics["is_terminal_step"]  # uneeded for logging

            return key, metrics

        # This loop is important because we don't want too many parallel envs.
        # So in evaluation we have num_envs parallel envs and loop enough times
        # so that we do at least `eval_episodes` number of episodes
        _, metrics = jax.lax.scan(_episode, key, xs=None, length=episode_loops)
        metrics: Metrics = jax.tree_map(lambda x: x.reshape(-1), metrics)  # flatten metrics
        return metrics

    return jax.pmap(eval_fn, axis_name="device")  # type: ignore


def make_ff_eval_act_fn(actor_network: nn.Module, config: DictConfig) -> EvalActFn:
    """Makes an act function that conforms to the evaluator API given a standard
    feed forward mava actor network."""

    def eval_act_fn(
        params: FrozenDict, timestep: TimeStep, key: PRNGKey, **_: ActorState
    ) -> Tuple[Action, Dict]:
        pi = actor_network.apply(params, timestep.observation)
        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
        return action, {}

    return eval_act_fn


def make_rec_eval_act_fn(actor_network: nn.Module, config: DictConfig) -> EvalActFn:
    """Makes an act function that conforms to the evaluator API given a standard
    recurrent mava actor network."""

    def eval_act_fn(
        params: FrozenDict, timestep: TimeStep, key: PRNGKey, **actor_state: ActorState
    ) -> Tuple[Action, Dict]:
        hidden_state = actor_state["hidden_state"]

        n_agents = timestep.observation.agents_view.shape[1]
        last_done = timestep.last()[:, jnp.newaxis].repeat(n_agents, axis=-1)
        ac_in = (timestep.observation, last_done)
        ac_in = jax.tree_map(lambda x: x[jnp.newaxis], ac_in)  # add batch dim to obs

        hidden_state, pi = actor_network.apply(params, hidden_state, ac_in)
        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
        return action.squeeze(0), {"hidden_state": hidden_state}

    return eval_act_fn  # type: ignore
