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

# Note this is only here until this is merged into jumanji
# PR: https://github.com/instadeepai/jumanji/pull/223

from typing import Tuple

import chex
import jax
from jumanji.env import State
from jumanji.types import TimeStep
from jumanji.wrappers import Observation, Wrapper

from mava.types import MarlEnv


class AutoResetWrapper(Wrapper):
    """Automatically resets environments that are done. Once the terminal state is reached,
    the state, observation, and step_type are reset. The observation and step_type of the
    terminal TimeStep is reset to the reset observation and StepType.LAST, respectively.
    The reward, discount, and extras retrieved from the transition to the terminal state.
    NOTE: The observation from the terminal TimeStep is stored in timestep.extras["real_next_obs"].
    WARNING: do not `jax.vmap` the wrapped environment (e.g. do not use with the `VmapWrapper`),
    which would lead to inefficient computation due to both the `step` and `reset` functions
    being processed each time `step` is called. Please use the `VmapAutoResetWrapper` instead.
    """

    OBS_IN_EXTRAS_KEY = "real_next_obs"

    # This init isn't really needed as jumanji.Wrapper will forward the attributes,
    # but mypy doesn't realize this.
    def __init__(self, env: MarlEnv):
        super().__init__(env)
        self._env: MarlEnv

        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self.action_dim = self._env.action_dim

    def _obs_in_extras(
        self, state: State, timestep: TimeStep[Observation]
    ) -> Tuple[State, TimeStep[Observation]]:
        """Place the observation in timestep.extras[real_next_obs]."""
        extras = timestep.extras
        extras[AutoResetWrapper.OBS_IN_EXTRAS_KEY] = timestep.observation
        return state, timestep.replace(extras=extras)

    def _auto_reset(
        self, state: State, timestep: TimeStep[Observation]
    ) -> Tuple[State, TimeStep[Observation]]:
        """Reset the state and overwrite `timestep.observation` with the reset observation
        if the episode has terminated.
        """
        if not hasattr(state, "key"):
            raise AttributeError(
                "This wrapper assumes that the state has attribute key which is used"
                " as the source of randomness for automatic reset"
            )

        # Make sure that the random key in the environment changes at each call to reset.
        # State is a type variable hence it does not have key type hinted, so we type ignore.
        key, _ = jax.random.split(state.key)  # type: ignore
        state, reset_timestep = self._env.reset(key)

        # Place original observation in extras.
        state, timestep = self._obs_in_extras(state, timestep)

        # Replace observation with reset observation.
        timestep = timestep.replace(observation=reset_timestep.observation)  # type: ignore

        return state, timestep

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep[Observation]]:
        return self._obs_in_extras(*super().reset(key))

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep[Observation]]:
        """Step the environment, with automatic resetting if the episode terminates."""
        state, timestep = self._env.step(state, action)

        # Overwrite the state and timestep appropriately if the episode terminates.
        state, timestep = jax.lax.cond(
            timestep.last(),
            self._auto_reset,
            self._obs_in_extras,
            state,
            timestep,
        )

        return state, timestep
