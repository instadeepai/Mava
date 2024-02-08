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

from typing import TYPE_CHECKING, Tuple

import chex
import jax.numpy as jnp
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import State

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class RecordEpisodeMetricsState:
    """State of the `LogWrapper`."""

    env_state: State
    # Temporary variables to keep track of the episode return and length.
    running_count_episode_return: chex.Numeric
    running_count_episode_length: chex.Numeric
    # Final episode return and length.
    episode_return: chex.Numeric
    episode_length: chex.Numeric


class RecordEpisodeMetrics(Wrapper):
    """Record the episode returns and lengths."""

    def reset(self, key: chex.PRNGKey) -> Tuple[RecordEpisodeMetricsState, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        state = RecordEpisodeMetricsState(state, 0.0, 0, 0.0, 0)
        timestep.extras["episode_metrics"] = {"episode_return": 0.0, "episode_length": 0}
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
        }

        state = RecordEpisodeMetricsState(
            env_state=env_state,
            running_count_episode_return=new_episode_return * not_done,
            running_count_episode_length=new_episode_length * not_done,
            episode_return=episode_return_info,
            episode_length=episode_length_info,
        )
        return state, timestep
