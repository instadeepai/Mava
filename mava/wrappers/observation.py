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

from typing import Tuple, Union

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import Observation, ObservationGlobalState, State


class AgentIDWrapper(Wrapper):
    """Add onehot agent IDs to observation."""

    def __init__(self, env: Environment, has_global_state: bool = False):
        super().__init__(env)
        self.has_global_state = has_global_state

    def _add_agent_ids(
        self, timestep: TimeStep, num_agents: int
    ) -> Union[Observation, ObservationGlobalState]:
        agent_ids = jnp.eye(num_agents)
        new_agents_view = jnp.concatenate([agent_ids, timestep.observation.agents_view], axis=-1)

        if self.has_global_state:
            # Add the agent IDs to the global state
            new_global_state = jnp.concatenate(
                [agent_ids, timestep.observation.global_state], axis=-1
            )

            return ObservationGlobalState(
                agents_view=new_agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
                global_state=new_global_state,
            )

        else:
            return Observation(
                agents_view=new_agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )

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

    def observation_spec(
        self,
    ) -> Union[specs.Spec[Observation], specs.Spec[ObservationGlobalState]]:
        """Specification of the observation of the `RobotWarehouse` environment."""
        obs_spec = self._env.observation_spec()
        num_obs_features = obs_spec.agents_view.shape[-1] + self._env.num_agents
        dtype = obs_spec.agents_view.dtype

        agents_view = specs.Array((self._env.num_agents, num_obs_features), dtype, "agents_view")

        if self.has_global_state:
            wrapped_state_shape = obs_spec.global_state.shape
            state_shape = (
                *wrapped_state_shape[:-1],
                wrapped_state_shape[-1] + self._env.num_agents,
            )
            global_state = specs.Array(state_shape, dtype, "global_state")
            return obs_spec.replace(agents_view=agents_view, global_state=global_state)

        return obs_spec.replace(agents_view=agents_view)


class GlobalStateWrapper(Wrapper):
    """Wrapper for adding global state to an environment that follows the mava API.

    The wrapper includes a global environment state to be used by the centralised critic.
    Note here that since most environments do not have a global state, we create one
    by concatenating the observations of all agents.
    """

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[ObservationGlobalState]:
        global_state = jnp.concatenate(timestep.observation.agents_view, axis=0)
        global_state = jnp.tile(global_state, (self._env.num_agents, 1))

        observation = ObservationGlobalState(
            global_state=global_state,
            agents_view=timestep.observation.agents_view,
            action_mask=timestep.observation.action_mask,
            step_count=timestep.observation.step_count,
        )

        return timestep.replace(observation=observation)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment. Updates the step count."""
        state, timestep = self._env.reset(key)
        return state, self.modify_timestep(timestep)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment. Updates the step count."""
        state, timestep = self._env.step(state, action)
        return state, self.modify_timestep(timestep)

    def observation_spec(self) -> specs.Spec[ObservationGlobalState]:
        """Specification of the observation of the `RobotWarehouse` environment."""

        obs_spec = self._env.observation_spec()
        num_obs_features = obs_spec.agents_view.shape[-1]
        global_state = specs.Array(
            (self._env.num_agents, self._env.num_agents * num_obs_features),
            obs_spec.agents_view.dtype,
            "global_state",
        )

        return specs.Spec(
            ObservationGlobalState,
            "ObservationSpec",
            agents_view=obs_spec.agents_view,
            action_mask=obs_spec.action_mask,
            global_state=global_state,
            step_count=obs_spec.step_count,
        )
