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

from functools import cached_property
from typing import Tuple, Union

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import MarlEnv, Observation, ObservationGlobalState, State


class AgentIDWrapper(Wrapper):
    """A wrapper to add a one-hot vector as agent IDs to the original observation.
    It can be useful in multi-agent environments where agents require unique identification.
    """

    # This init isn't really needed as jumanji.Wrapper will forward the attributes,
    # but mypy doesn't realize this.
    def __init__(self, env: MarlEnv):
        super().__init__(env)
        self._env: MarlEnv

        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self.action_dim = self._env.action_dim

    def _add_agent_ids(
        self, timestep: TimeStep, num_agents: int
    ) -> Union[Observation, ObservationGlobalState]:
        """Adds agent IDs to the observation."""
        obs = timestep.observation
        agent_ids = jnp.eye(num_agents)
        agents_view = jnp.concatenate(
            [agent_ids, obs.agents_view],
            axis=-1,
            dtype=obs.agents_view.dtype,
        )

        return obs._replace(agents_view=agents_view)  # type: ignore

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        timestep.observation = self._add_agent_ids(timestep, self._env.num_agents)

        return state, timestep

    def step(
        self,
        state: State,
        action: chex.Array,
    ) -> Tuple[State, TimeStep]:
        """Step the environment."""
        state, timestep = self._env.step(state, action)
        timestep.observation = self._add_agent_ids(timestep, self._env.num_agents)

        return state, timestep

    @cached_property
    def observation_spec(
        self,
    ) -> Union[specs.Spec[Observation], specs.Spec[ObservationGlobalState]]:
        """Specification of the observation of the selected environment."""
        obs_spec = self._env.observation_spec
        num_obs_features = obs_spec.agents_view.shape[-1] + self._env.num_agents
        dtype = obs_spec.agents_view.dtype
        agents_view = specs.Array((self._env.num_agents, num_obs_features), dtype, "agents_view")

        return obs_spec.replace(agents_view=agents_view)
