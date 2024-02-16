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

from typing import Tuple

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import Observation, State


class MatraxWrapper(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)
        self._num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit
        self._num_actions = self._env.num_actions
        self.num_obs_features = self._env.observation_spec().agent_obs.shape[-1]

    def modify_timestep_step(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for `step` and `reset`."""
        observation = Observation(
            agents_view=timestep.observation.agent_obs,
            action_mask=jnp.ones((self._num_agents, self._num_actions), dtype=bool),
            step_count=jnp.repeat(timestep.observation.step_count, self._num_agents),
        )
        return timestep.replace(
            observation=observation,
            reward=timestep.reward * 1.0,
            discount=timestep.discount,
        )

    def modify_timestep_reset(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for `step` and `reset`."""
        observation = Observation(
            agents_view=timestep.observation.agent_obs,
            action_mask=jnp.ones((self._num_agents, self._num_actions), dtype=bool),
            step_count=jnp.repeat(timestep.observation.step_count, self._num_agents),
        )
        reward = jnp.repeat(timestep.reward, self._num_agents)
        return timestep.replace(
            observation=observation,
            reward=reward,
            discount=timestep.discount,
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        return state, self.modify_timestep_reset(timestep)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment."""
        state, timestep = self._env.step(state, action)
        return state, self.modify_timestep_step(timestep)

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the environment."""
        env_observation_spec = self._env.observation_spec().agent_obs
        step_count = specs.BoundedArray(
            (self._num_agents,),
            jnp.int32,
            [0] * self._num_agents,
            [self.time_limit] * self._num_agents,
            "step_count",
        )
        action_mask = specs.Array(
            (self._num_agents, self._num_actions),
            jnp.bool_,
            "action_mask",
        )
        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=env_observation_spec,
            action_mask=action_mask,
            step_count=step_count,
        )
