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

from mava.types import Action, EvalFn, Observation, ObservationGlobalState

# Optional extras that are passed out of the actor and then into the actor in the next step
ActorState: TypeAlias = Dict[str, Any]


class EvalActFn(Protocol):
    def __call__(
        self,
        params: FrozenDict,
        timestep: TimeStep[Union[Observation, ObservationGlobalState]],
        key: PRNGKey,
        **actor_state,
    ) -> Tuple[Array, ActorState]: ...


def get_eval_fn(
    env: Environment, act_fn: EvalActFn, config: DictConfig, eval_episodes: int
) -> EvalFn:
    def eval_fn(
        params: FrozenDict, key: PRNGKey, init_act_state: ActorState = {}
    ) -> Dict[str, Array]:
        def _env_step(eval_state, actor_state):
            """Performs a single environment step"""
            env_state, ts, key, actor_state = eval_state

            key, act_key = jax.random.split(key)
            action, actor_state = act_fn(params, ts, act_key, **actor_state)
            env_state, ts = jax.vmap(env.step)(env_state, action)

            return (env_state, ts, key, actor_state), ts

        def _episode(key: PRNGKey, _) -> Tuple[PRNGKey, Dict[str, Array]]:
            """Simulates `config.arch.num_envs` episodes."""
            key, reset_key = jax.random.split(key)
            reset_keys = jax.random.split(reset_key, config.arch.num_envs)
            env_state, ts = jax.vmap(env.reset)(reset_keys)

            step_state = env_state, ts, key, init_act_state
            _, timesteps = jax.lax.scan(_env_step, step_state, jnp.arange(env.time_limit))

            metrics = timesteps.extras["episode_metrics"]
            if config.env.log_win_rate:
                metrics["won_episode"] = timesteps.extras["won_episode"]

            # find the first instance of done to get the metrics at that timestep
            # we don't care about subsequent steps because we only the results from the first episode
            done_idx = jnp.argmax(timesteps.last(), axis=0)
            metrics = jax.tree_map(lambda m: m[done_idx, jnp.arange(config.arch.num_envs)], metrics)
            del metrics["is_terminal_step"]  # uneeded for logging

            return key, metrics

        # todo: do we want to divide this by n devices?
        episode_loops = math.ceil(eval_episodes / config.arch.num_envs)
        # This loop is important because we don't want too many parallel envs.
        # So in evaluation we have config.arch.num_envs parallel envs and loop
        # enough times so that we do at least `eval_episodes` number of episodes
        _, metrics = jax.lax.scan(_episode, key, xs=None, length=episode_loops)
        # todo: can we calculate sps here?
        return jax.tree_map(lambda x: x.reshape(-1), metrics)  # flatten metrics

    return jax.pmap(eval_fn, axis_name="device")


def make_ff_eval_act_fn(actor_network: nn.Module, config: DictConfig) -> EvalActFn:
    """Makes an act function that works with the evaluator API given a standard feed forward mava actor network."""

    def eval_act_fn(params, timestep, key, **_) -> Tuple[Action, Dict]:
        pi = actor_network.apply(params, timestep.observation)
        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
        return action, {}

    return eval_act_fn


def make_rec_eval_act_fn(actor_network: nn.Module, config: DictConfig) -> EvalActFn:
    """Makes an act function that works with the evaluator API given a standard recurrent mava actor network."""

    def eval_act_fn(params, timestep, key, hidden_state) -> Tuple[Action, Dict]:
        n_agents = timestep.observation.agents_view.shape[1]
        last_done = timestep.last()[:, jnp.newaxis].repeat(n_agents, axis=-1)
        ac_in = (timestep.observation, last_done)
        ac_in = jax.tree_map(lambda x: x[jnp.newaxis], ac_in)  # add batch dim to obs

        hidden_state, pi = actor_network.apply(params, hidden_state, ac_in)
        action = pi.mode() if config.arch.evaluation_greedy else pi.sample(seed=key)
        return action.squeeze(0), {"hidden_state": hidden_state}

    return eval_act_fn
