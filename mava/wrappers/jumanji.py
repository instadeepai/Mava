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

from abc import abstractmethod
from functools import cached_property
from typing import Tuple, Union

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.connector import MaConnector
from jumanji.environments.routing.connector.constants import (
    EMPTY,
    PATH,
    POSITION,
    TARGET,
)
from jumanji.environments.routing.lbf import LevelBasedForaging
from jumanji.environments.routing.robot_warehouse import RobotWarehouse
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import Observation, ObservationGlobalState, State


class MultiAgentWrapper(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)
        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for `step` and `reset`."""
        pass

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        return state, self.modify_timestep(timestep)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment."""
        state, timestep = self._env.step(state, action)
        return state, self.modify_timestep(timestep)

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self._num_agents,),
            int,
            jnp.zeros(self._num_agents, dtype=int),
            jnp.repeat(self.time_limit, self._num_agents),
            "step_count",
        )
        return self._env.observation_spec().replace(step_count=step_count)

    @cached_property
    @abstractmethod
    def min_max_action(self) -> Tuple[chex.Array, chex.Array]:
        """Returns the minimum and maximum values from the action space"""
        ...

    @cached_property
    @abstractmethod
    def action_dim(self) -> chex.Array:
        "Get the actions dim for each agent."
        ...


class RwareWrapper(MultiAgentWrapper):
    """Multi-agent wrapper for the Robotic Warehouse environment."""

    def __init__(self, env: RobotWarehouse):
        super().__init__(env)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for the Robotic Warehouse environment."""
        observation = Observation(
            agents_view=timestep.observation.agents_view,
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self.num_agents),
        )
        reward = jnp.repeat(timestep.reward, self.num_agents)
        discount = jnp.repeat(timestep.discount, self.num_agents)
        return timestep.replace(observation=observation, reward=reward, discount=discount)

    @cached_property
    def action_dim(self) -> chex.Array:
        "Get the actions dim for each agent."
        return int(self._env.action_spec().num_values[0])


class LbfWrapper(MultiAgentWrapper):
    """
     Multi-agent wrapper for the Level-Based Foraging environment.

    Args:
        env (Environment): The base environment.
        use_individual_rewards (bool): If true each agent gets a separate reward,
        sum reward otherwise.
    """

    def __init__(self, env: LevelBasedForaging, use_individual_rewards: bool = False):
        super().__init__(env)
        self._env: LevelBasedForaging
        self._use_individual_rewards = use_individual_rewards

    def aggregate_rewards(
        self, timestep: TimeStep, observation: Observation
    ) -> TimeStep[Observation]:
        """Aggregate individual rewards across agents."""
        team_reward = jnp.sum(timestep.reward)

        # Repeat the aggregated reward for each agent.
        reward = jnp.repeat(team_reward, self.num_agents)
        return timestep.replace(observation=observation, reward=reward)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for Level-Based Foraging environment and update
        the reward based on the specified reward handling strategy."""

        # Create a new observation with adjusted step count
        modified_observation = Observation(
            agents_view=timestep.observation.agents_view,
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self.num_agents),
        )
        if self._use_individual_rewards:
            # The environment returns a list of individual rewards and these are used as is.
            return timestep.replace(observation=modified_observation)

        # Aggregate the list of individual rewards and use a single team_reward.
        return self.aggregate_rewards(timestep, modified_observation)

    @cached_property
    def action_dim(self) -> chex.Array:
        "Get the actions dim for each agent."
        return int(self._env.action_spec().num_values[0])


class ConnectorWrapper(MultiAgentWrapper):
    """Multi-agent wrapper for the MA Connector environment.

    Do not use the AgentID wrapper with this env, it has implicit agent IDs.
    """

    def __init__(self, env: MaConnector, has_global_state: bool = False):
        super().__init__(env)
        self.has_global_state = has_global_state

    def modify_timestep(
        self, timestep: TimeStep
    ) -> TimeStep[Union[Observation, ObservationGlobalState]]:
        """Modify the timestep for the Connector environment."""

        # TARGET = 3 = The number of different types of items on the grid.
        def create_agents_view(grid: chex.Array) -> chex.Array:
            positions = jnp.where(grid % TARGET == POSITION, True, False)
            targets = jnp.where((grid % TARGET == 0) & (grid != EMPTY), True, False)
            paths = jnp.where(grid % TARGET == PATH, True, False)
            position_per_agent = jnp.where(grid == POSITION, True, False)
            target_per_agent = jnp.where(grid == TARGET, True, False)
            agents_view = jnp.stack(
                (positions, targets, paths, position_per_agent, target_per_agent), -1
            )
            return agents_view

        def create_global_state(grid: chex.Array) -> chex.Array:
            positions = jnp.where(grid % TARGET == POSITION, True, False)
            targets = jnp.where((grid % TARGET == 0) & (grid != EMPTY), True, False)
            paths = jnp.where(grid % TARGET == PATH, True, False)
            global_state = jnp.stack((positions, targets, paths), -1)
            return global_state

        obs_data = {
            "agents_view": create_agents_view(timestep.observation.grid),
            "action_mask": timestep.observation.action_mask,
            "step_count": jnp.repeat(timestep.observation.step_count, self._num_agents),
        }

        if self.has_global_state:
            obs_data["global_state"] = create_global_state(timestep.observation.grid)
            return timestep.replace(observation=ObservationGlobalState(**obs_data))
        else:
            return timestep.replace(observation=Observation(**obs_data))

    def observation_spec(self) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self._num_agents,),
            int,
            jnp.zeros(self._num_agents, dtype=int),
            jnp.repeat(self.time_limit, self._num_agents),
            "step_count",
        )

        agents_view = specs.BoundedArray(
            shape=(self._env.num_agents, self._env.grid_size, self._env.grid_size, 5),
            dtype=bool,
            name="agents_view",
            minimum=False,
            maximum=True,
        )

        if self.has_global_state:
            global_state = specs.BoundedArray(
                shape=(self._env.num_agents, self._env.grid_size, self._env.grid_size, 3),
                dtype=bool,
                name="global_state",
                minimum=False,
                maximum=True,
            )
            spec = specs.Spec(
                ObservationGlobalState,
                "ObservationSpec",
                agents_view=agents_view,
                action_mask=self._env.observation_spec().action_mask,
                global_state=global_state,
                step_count=step_count,
            )

        else:
            spec = specs.Spec(
                Observation,
                "ObservationSpec",
                agents_view=agents_view,
                action_mask=self._env.observation_spec().action_mask,
                step_count=step_count,
            )

        return spec
