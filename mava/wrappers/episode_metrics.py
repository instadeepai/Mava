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

from typing import TYPE_CHECKING, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax import tree
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import MarlEnv, State

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class RecordEpisodeMetricsState:
    """State of the `LogWrapper`."""

    env_state: State
    key: chex.PRNGKey
    # Temporary variables to keep track of the episode return and length.
    running_count_episode_return: chex.Numeric
    running_count_episode_length: chex.Numeric
    # Final episode return and length.
    episode_return: chex.Numeric
    episode_length: chex.Numeric


class RecordEpisodeMetrics(Wrapper):
    """Record the episode returns and lengths."""

    # This init isn't really needed as jumanji.Wrapper will forward the attributes,
    # but mypy doesn't realize this.
    def __init__(self, env: MarlEnv):
        super().__init__(env)
        self._env: MarlEnv

        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self.action_dim = self._env.action_dim

    def reset(self, key: chex.PRNGKey) -> Tuple[RecordEpisodeMetricsState, TimeStep]:
        """Reset the environment."""
        key, reset_key = jax.random.split(key)
        state, timestep = self._env.reset(reset_key)
        state = RecordEpisodeMetricsState(
            state,
            key,
            jnp.array(0.0, dtype=float),
            jnp.array(0, dtype=int),
            jnp.array(0.0, dtype=float),
            jnp.array(0, dtype=int),
        )
        timestep.extras["episode_metrics"] = {
            "episode_return": jnp.array(0.0, dtype=float),
            "episode_length": jnp.array(0, dtype=int),
            "is_terminal_step": jnp.array(False, dtype=bool),
        }
        return state, timestep

    def step(
        self,
        state: RecordEpisodeMetricsState,
        action: chex.Array,
    ) -> Tuple[RecordEpisodeMetricsState, TimeStep]:
        """Step the environment."""
        env_state, timestep = self._env.step(state.env_state, action)

        done = timestep.last()
        not_done = 1 - done

        # Counting episode return and length.
        new_episode_return = state.running_count_episode_return + jnp.mean(timestep.reward)
        new_episode_length = state.running_count_episode_length + 1

        # Previous episode return/length until done and then the next episode return.
        episode_return_info = state.episode_return * not_done + new_episode_return * done
        episode_length_info = state.episode_length * not_done + new_episode_length * done

        timestep.extras["episode_metrics"] = {
            "episode_return": episode_return_info,
            "episode_length": episode_length_info,
            "is_terminal_step": done,
        }

        state = RecordEpisodeMetricsState(
            env_state=env_state,
            key=state.key,
            running_count_episode_return=new_episode_return * not_done,
            running_count_episode_length=new_episode_length * not_done,
            episode_return=episode_return_info,
            episode_length=episode_length_info,
        )
        return state, timestep


def get_final_step_metrics(metrics: Dict[str, chex.Array]) -> Tuple[Dict[str, chex.Array], bool]:
    """Get the metrics for the final step of an episode and check if there was a final step
    within the provided metrics.

    Note: this is not a jittable method. We need to return variable length arrays, since
    we don't know how many episodes have been run. This is done since the logger
    expects arrays for computing summary statistics on the episode metrics.
    """
    is_final_ep = metrics.pop("is_terminal_step")
    has_final_ep_step = bool(np.any(is_final_ep))

    final_metrics: Dict[str, chex.Array]
    # If it didn't make it to the final step, return zeros.
    if not has_final_ep_step:
        final_metrics = tree.map(np.zeros_like, metrics)
    else:
        final_metrics = tree.map(lambda x: x[is_final_ep], metrics)

    return final_metrics, has_final_ep_step
