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
from jumanji.env import Environment
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper
from matrax.env import MatrixGame

from mava.types import Observation, ObservationGlobalState, State


class MatraxWrapper(Wrapper):
    """Multi-agent wrapper for the Matrax environment."""

    def __init__(self, env: Environment, add_global_state: bool):
        self.add_global_state = add_global_state
        super().__init__(env)
        self._env: MatrixGame

        self.num_agents = self._env.num_agents
        self.action_dim = self._env.num_actions
        self.time_limit = self._env.time_limit
        self.action_mask = jnp.ones((self.num_agents, self.num_actions), dtype=bool)

    def modify_timestep(
        self, timestep: TimeStep
    ) -> TimeStep[Union[Observation, ObservationGlobalState]]:
        """Modify the timestep for `step` and `reset`."""
        obs_data = {
            "agents_view": timestep.observation.agent_obs,
            "action_mask": self.action_mask,
            "step_count": jnp.repeat(timestep.observation.step_count, self.num_agents),
        }
        if self.add_global_state:
            global_state = jnp.concatenate(timestep.observation.agent_obs, axis=0)
            global_state = jnp.tile(global_state, (self.num_agents, 1))
            obs_data["global_state"] = global_state
            return timestep.replace(observation=ObservationGlobalState(**obs_data))

        return timestep.replace(observation=Observation(**obs_data))

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        return state, self.modify_timestep(timestep)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment."""
        state, timestep = self._env.step(state, action)
        return state, self.modify_timestep(timestep)

    @cached_property
    def observation_spec(self) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self.num_agents,),
            int,
            jnp.zeros(self.num_agents, dtype=int),
            jnp.repeat(self.time_limit, self.num_agents),
            "step_count",
        )
        action_mask = specs.Array(
            (self.num_agents, self.num_actions),
            bool,
            "action_mask",
        )
        obs_spec = self._env.observation_spec
        obs_data = {
            "agents_view": obs_spec.agent_obs,
            "action_mask": action_mask,
            "step_count": step_count,
        }
        if self.add_global_state:
            num_obs_features = obs_spec.agent_obs.shape[-1]
            global_state = specs.Array(
                (self._env.num_agents, self._env.num_agents * num_obs_features),
                obs_spec.agent_obs.dtype,
                "global_state",
            )
            obs_data["global_state"] = global_state
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)

        return specs.Spec(Observation, "ObservationSpec", **obs_data)
