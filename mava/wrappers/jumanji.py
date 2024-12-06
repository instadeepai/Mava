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


from abc import ABC, abstractmethod
from functools import cached_property
from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.environments.routing.cleaner import Cleaner
from jumanji.environments.routing.cleaner.constants import DIRTY, WALL
from jumanji.environments.routing.connector import Connector
from jumanji.environments.routing.connector.constants import (
    AGENT_INITIAL_VALUE,
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


def aggregate_rewards(reward: chex.Array, num_agents: int) -> chex.Array:
    """Aggregate individual rewards across agents."""
    team_reward = jnp.sum(reward)
    return jnp.repeat(team_reward, num_agents)


class JumanjiMarlWrapper(Wrapper, ABC):
    def __init__(self, env: Environment, add_global_state: bool):
        self.add_global_state = add_global_state
        super().__init__(env)
        self.num_agents = self._env.num_agents
        self.time_limit = self._env.time_limit

    @abstractmethod
    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for `step` and `reset`."""
        pass

    def get_global_state(self, obs: Observation) -> chex.Array:
        """The default way to create a global state for an environment if it has no
        available global state - concatenate all observations.
        """
        global_state = jnp.concatenate(obs.agents_view, axis=0)
        global_state = jnp.tile(global_state, (self._env.num_agents, 1))
        return global_state

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        timestep = self.modify_timestep(timestep)
        if self.add_global_state:
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return state, timestep.replace(observation=observation)

        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment."""
        state, timestep = self._env.step(state, action)
        timestep = self.modify_timestep(timestep)
        if self.add_global_state:
            global_state = self.get_global_state(timestep.observation)
            observation = ObservationGlobalState(
                global_state=global_state,
                agents_view=timestep.observation.agents_view,
                action_mask=timestep.observation.action_mask,
                step_count=timestep.observation.step_count,
            )
            return state, timestep.replace(observation=observation)

        return state, timestep

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

        obs_spec = self._env.observation_spec
        obs_data = {
            "agents_view": obs_spec.agents_view,
            "action_mask": obs_spec.action_mask,
            "step_count": step_count,
        }

        if self.add_global_state:
            num_obs_features = obs_spec.agents_view.shape[-1]
            global_state = specs.Array(
                (self._env.num_agents, self._env.num_agents * num_obs_features),
                obs_spec.agents_view.dtype,
                "global_state",
            )
            obs_data["global_state"] = global_state
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)

        return specs.Spec(Observation, "ObservationSpec", **obs_data)

    @cached_property
    def action_dim(self) -> chex.Array:
        """Get the actions dim for each agent."""
        return int(self._env.action_spec.num_values[0])


class RwareWrapper(JumanjiMarlWrapper):
    """Multi-agent wrapper for the Robotic Warehouse environment."""

    def __init__(self, env: RobotWarehouse, add_global_state: bool = False):
        super().__init__(env, add_global_state)
        self._env: RobotWarehouse

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for the Robotic Warehouse environment."""
        observation = Observation(
            agents_view=timestep.observation.agents_view.astype(float),
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self.num_agents),
        )
        reward = jnp.repeat(timestep.reward, self.num_agents)
        discount = jnp.repeat(timestep.discount, self.num_agents)
        return timestep.replace(observation=observation, reward=reward, discount=discount)

    @cached_property
    def observation_spec(
        self,
    ) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        # need to cast the agents view and global state to floats as we do in modify timestep
        inner_spec = super().observation_spec
        spec = inner_spec.replace(agents_view=inner_spec.agents_view.replace(dtype=float))
        if self.add_global_state:
            spec = spec.replace(global_state=inner_spec.global_state.replace(dtype=float))

        return spec


class LbfWrapper(JumanjiMarlWrapper):
    """Multi-agent wrapper for the Level-Based Foraging environment.

    Args:
    ----
        env (Environment): The base environment.
        use_individual_rewards (bool): If true each agent gets a separate reward,
        sum reward otherwise.

    """

    def __init__(
        self,
        env: LevelBasedForaging,
        add_global_state: bool = False,
        aggregate_rewards: bool = True,
    ):
        super().__init__(env, add_global_state)
        self._env: LevelBasedForaging
        self._aggregate_rewards = aggregate_rewards

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for Level-Based Foraging environment and update
        the reward based on the specified reward handling strategy.
        """
        # Create a new observation with adjusted step count
        modified_observation = Observation(
            agents_view=timestep.observation.agents_view.astype(float),
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self.num_agents),
        )
        # Whether or not aggregate the list of individual rewards.
        reward = timestep.reward
        if self._aggregate_rewards:
            reward = aggregate_rewards(reward, self.num_agents)

        return timestep.replace(observation=modified_observation, reward=reward)

    @cached_property
    def observation_spec(
        self,
    ) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        # need to cast the agents view and global state to floats as we do in modify timestep
        inner_spec = super().observation_spec
        spec = inner_spec.replace(agents_view=inner_spec.agents_view.replace(dtype=float))
        if self.add_global_state:
            spec = spec.replace(global_state=inner_spec.global_state.replace(dtype=float))

        return spec


def switch_perspective(grid: chex.Array, agent_id: int, num_agents: int) -> chex.Array:
    """
    Encodes the observation with respect to the current agent defined by `agent_id`.
    Each agent sees its observations as values `1, 2, 3`. Observations of other agents
    are shifted cyclically based on their relative position. The mapping is designed
    such that the ordering of observations remains consistent.
    For example,in a 3-agent game, if we wanted to switch to agent 1's perspective, then:
    agent 1s values will change from 4,5,6 -> 1,2,3
    agent 2s values will change from 7,8,9 -> 4,5,6
    agent 0s values will change from 1,2,3 -> 7,8,9
    Agent 0 will be passed observations where it is represented by the values 1,2,3. Agent 1
    will be passed observations where it is represented by the values 1,2,3. However in the
    state agent 0 will always be 1,2,3 and agent 1 will always be 4,5,6."""
    new_grid = grid - AGENT_INITIAL_VALUE  # Center agent values around 0
    new_grid -= 3 * agent_id  # Move the obs
    new_grid %= 3 * num_agents  # Keep obs in bounds
    new_grid += AGENT_INITIAL_VALUE  # 'Un-center' agent obs around 0
    # Take agent values from rotated grid and empty values from old grid
    return jnp.where((grid >= AGENT_INITIAL_VALUE), new_grid, grid)


class ConnectorWrapper(JumanjiMarlWrapper):
    """Multi-agent wrapper for the MA Connector environment.

    Do not use the AgentID wrapper with this env, it has implicit agent IDs.
    """

    def __init__(
        self, env: Connector, add_global_state: bool = False, aggregate_rewards: bool = True
    ):
        super().__init__(env, add_global_state)
        self._env: Connector
        self._aggregate_rewards = aggregate_rewards
        self.agent_ids = jnp.arange(self.num_agents)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for the Connector environment."""

        # TARGET = 3 = The number of different types of items on the grid.
        def create_agents_view(grid: chex.Array) -> chex.Array:
            grid = jax.vmap(switch_perspective, in_axes=(None, 0, None))(
                grid, self.agent_ids, self.num_agents
            )
            # Mark position and target of each agent with that agent's normalized index.
            positions = (
                jnp.where(grid % TARGET == POSITION, jnp.ceil(grid / TARGET), 0) / self.num_agents
            )
            targets = (
                jnp.where((grid % TARGET == 0) & (grid != EMPTY), jnp.ceil(grid / TARGET), 0)
                / self.num_agents
            )
            paths = jnp.where(grid % TARGET == PATH, 1, 0)
            position_per_agent = jnp.where(grid == POSITION, 1, 0)
            target_per_agent = jnp.where(grid == TARGET, 1, 0)
            agents_view = jnp.stack(
                (positions, targets, paths, position_per_agent, target_per_agent), -1
            )
            return agents_view

        obs_data = {
            "agents_view": create_agents_view(timestep.observation.grid),
            "action_mask": timestep.observation.action_mask,
            "step_count": jnp.repeat(timestep.observation.step_count, self.num_agents),
        }

        # The episode is won if all agents have connected.
        extras = timestep.extras | {"won_episode": timestep.extras["ratio_connections"] == 1.0}

        # Whether or not aggregate the list of individual rewards.
        reward = timestep.reward
        if self._aggregate_rewards:
            reward = aggregate_rewards(reward, self.num_agents)
        return timestep.replace(observation=Observation(**obs_data), reward=reward, extras=extras)

    def get_global_state(self, obs: Observation) -> chex.Array:
        """Constructs the global state from the global information
        in the agent observations (positions, targets and paths.)
        """
        return jnp.tile(obs.agents_view[..., :3][0], (obs.agents_view.shape[0], 1, 1, 1))

    @cached_property
    def observation_spec(
        self,
    ) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self.num_agents,),
            int,
            jnp.zeros(self.num_agents, dtype=int),
            jnp.repeat(self.time_limit, self.num_agents),
            "step_count",
        )
        agents_view = specs.BoundedArray(
            shape=(self._env.num_agents, self._env.grid_size, self._env.grid_size, 5),
            dtype=float,
            name="agents_view",
            minimum=0.0,
            maximum=1.0,
        )
        obs_data = {
            "agents_view": agents_view,
            "action_mask": self._env.observation_spec.action_mask,
            "step_count": step_count,
        }
        if self.add_global_state:
            global_state = specs.BoundedArray(
                shape=(self._env.num_agents, self._env.grid_size, self._env.grid_size, 3),
                dtype=float,
                name="global_state",
                minimum=0.0,
                maximum=1.0,
            )
            obs_data["global_state"] = global_state
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)

        return specs.Spec(Observation, "ObservationSpec", **obs_data)


def _slice_around(pos: chex.Array, fov: int) -> Tuple[chex.Array, chex.Array]:
    """Return the start and length of a slice that when used to index a grid will
    return a 2*fov+1 x 2*fov+1 sub-grid centered around pos.

    Returns are meant to be used with a `jax.lax.dynamic_slice`
    """
    # Because we pad the grid by fov we need to shift the pos to the position
    # it will be in the padded grid.
    shifted_pos = pos + fov

    start_x = shifted_pos[0] - fov
    start_y = shifted_pos[1] - fov
    return start_x, start_y


# get location coordinates from 2D grid
def _get_location(grid: chex.Array) -> chex.Array:
    row_len = grid.shape[-1]
    index = jnp.argmax(grid)
    return jnp.asarray((jnp.floor(index / row_len), jnp.remainder(index, row_len)), dtype=int)


class VectorConnectorWrapper(JumanjiMarlWrapper):
    """Multi-agent wrapper for the Connector environment.

    This wrapper transforms the grid-based observation to a vector of features. This env should
    have the AgentID wrapper applied to it since there is not longer a channel that can encode
    AgentID information.
    """

    def __init__(
        self, env: Connector, add_global_state: bool = False, aggregate_rewards: bool = True
    ):
        self.fov = 2
        super().__init__(env, add_global_state)
        self._env: Connector
        self._aggregate_rewards = aggregate_rewards
        self.agent_ids = jnp.arange(self.num_agents)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for the Connector environment."""

        # TARGET = 3 = The number of different types of items on the grid.
        def create_agents_view(grid: chex.Array) -> chex.Array:
            grid = jax.vmap(switch_perspective, in_axes=(None, 0, None))(
                grid, self.agent_ids, self.num_agents
            )
            positions = jnp.where(grid % TARGET == POSITION, True, False)
            targets = jnp.where((grid % TARGET == 0) & (grid != EMPTY), True, False)
            paths = jnp.where(grid % TARGET == PATH, True, False)

            # group positions and paths
            blockers = jnp.where(positions, 1, jnp.where(paths, -1, 0))

            position_per_agent = grid == POSITION
            target_per_agent = grid == TARGET

            # group agents own target and other targets
            combined_targets = jnp.where(target_per_agent, 1, jnp.where(targets, -1, 0))

            # get coordinates of each agent's location and target
            position_coords = jax.vmap(_get_location)(position_per_agent)
            target_coords = jax.vmap(_get_location)(target_per_agent)

            def _create_one_agent_view(i: int) -> chex.Array:
                slice_len = 2 * self.fov + 1, 2 * self.fov + 1
                slice_x, slice_y = _slice_around(position_coords[i], self.fov)
                padded_blockers = jnp.pad(blockers[i], self.fov, constant_values=True)

                blockers_around_agent = jax.lax.dynamic_slice(
                    padded_blockers, (slice_x, slice_y), slice_len
                )
                blockers_around_agent = jnp.reshape(blockers_around_agent, -1).astype(float)

                my_pos = position_coords[i] / grid[0].size
                my_target = target_coords[i] / grid[0].size

                padded_combined_targets = jnp.pad(
                    combined_targets[i], self.fov, constant_values=True
                )

                targets_around_agent = jax.lax.dynamic_slice(
                    padded_combined_targets, (slice_x, slice_y), slice_len
                )
                targets_around_agent = jnp.reshape(targets_around_agent, -1).astype(float)

                return jnp.concatenate(
                    [my_pos, my_target, blockers_around_agent, targets_around_agent],
                    dtype=float,
                )

            return jax.vmap(_create_one_agent_view)(jnp.arange(self.num_agents))

        obs_data = {
            "agents_view": create_agents_view(timestep.observation.grid),
            "action_mask": timestep.observation.action_mask,
            "step_count": jnp.repeat(timestep.observation.step_count, self.num_agents),
        }

        # The episode is won if all agents have connected.
        extras = timestep.extras | {"won_episode": timestep.extras["ratio_connections"] == 1.0}

        reward = timestep.reward
        if self._aggregate_rewards:
            reward = aggregate_rewards(reward, self.num_agents)
        return timestep.replace(observation=Observation(**obs_data), reward=reward, extras=extras)

    @cached_property
    def observation_spec(
        self,
    ) -> specs.Spec[Union[Observation, ObservationGlobalState]]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self.num_agents,),
            int,
            jnp.zeros(self.num_agents, dtype=int),
            jnp.repeat(self.time_limit, self.num_agents),
            "step_count",
        )
        # 2 sets of tiles in fov (blockers and targets) + xy position of agent and target
        tiles_in_fov = (self.fov * 2 + 1) ** 2
        single_agent_obs = 4 + tiles_in_fov * 2
        agents_view = specs.BoundedArray(
            shape=(self.num_agents, single_agent_obs),
            dtype=float,
            name="agents_view",
            minimum=-1.0,
            maximum=1.0,
        )
        obs_data = {
            "agents_view": agents_view,
            "action_mask": self._env.observation_spec.action_mask,
            "step_count": step_count,
        }
        if self.add_global_state:
            global_state = specs.BoundedArray(
                shape=(self.num_agents, self.num_agents * single_agent_obs),
                dtype=float,
                name="global_state",
                minimum=-1.0,
                maximum=1.0,
            )
            obs_data["global_state"] = global_state
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)

        return specs.Spec(Observation, "ObservationSpec", **obs_data)


class CleanerWrapper(JumanjiMarlWrapper):
    """Multi-agent wrapper for the Cleaner environment."""

    def __init__(self, env: Cleaner, add_global_state: bool = False):
        super().__init__(env, add_global_state)
        self._env: Cleaner

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep for the Cleaner environment."""

        def create_agents_view(grid: chex.Array, agents_locations: chex.Array) -> chex.Array:
            """Create separate channels for dirty cells, wall cells and agent positions.
            Also add a channel that marks an agent's own position.
            """
            num_agents = self.num_agents

            # A: Number of agents
            # R: Number of grid rows
            # C: Number of grid columns
            # grid: (R, C)
            # agents_locations: (A, 2)

            # Get dirty / wall tiles from first agent's obs and tile in agents dimension.

            dirty_channel = jnp.tile(grid == DIRTY, (num_agents, 1, 1))  # (A, R, C)
            wall_channel = jnp.tile(grid == WALL, (num_agents, 1, 1))  # (A, R, C)

            # Get each agent's position.
            xs, ys = agents_locations[:, 0], agents_locations[:, 1]  # (A,), (A,)

            # Mask each agent's position so an agent can idenfity itself.
            # Sum the masked grids together for global agent information.
            # (A, R, C)
            pos_per_agent = jnp.repeat(jnp.zeros_like(grid)[jnp.newaxis, :, :], num_agents, axis=0)
            pos_per_agent = pos_per_agent.at[jnp.arange(num_agents), xs, ys].set(1)  # (A, R, C)
            # (A, R, C)
            agents_channel = jnp.tile(jnp.sum(pos_per_agent, axis=0), (num_agents, 1, 1))

            # Stack the channels along the last dimension.
            agents_view = jnp.stack(
                [dirty_channel, wall_channel, agents_channel, pos_per_agent],
                axis=-1,  # (A, R, C, 4)
            )
            return agents_view

        obs_data = {
            "agents_view": create_agents_view(
                timestep.observation.grid, timestep.observation.agents_locations
            ),
            "action_mask": timestep.observation.action_mask,
            "step_count": jnp.repeat(timestep.observation.step_count, self.num_agents),
        }

        reward = jnp.repeat(timestep.reward, self.num_agents)
        discount = jnp.repeat(timestep.discount, self.num_agents)

        # The episode is won if every tile is cleaned.
        extras = {"won_episode": timestep.extras["num_dirty_tiles"] == 0}

        return timestep.replace(
            observation=Observation(**obs_data), reward=reward, discount=discount, extras=extras
        )

    def get_global_state(self, obs: Observation) -> chex.Array:
        """Constructs the global state from the global information
        in the agent observations (dirty tiles, wall tiles and agent positions).
        """
        return obs.agents_view[..., :3]  # (A, R, C, 3)

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
        agents_view = specs.BoundedArray(
            shape=(self.num_agents, self._env.num_rows, self._env.num_cols, 4),
            dtype=bool,
            name="agents_view",
            minimum=0,
            maximum=self.num_agents,
        )
        obs_data = {
            "agents_view": agents_view,
            "action_mask": self._env.observation_spec.action_mask,
            "step_count": step_count,
        }
        if self.add_global_state:
            global_state = specs.BoundedArray(
                shape=(self.num_agents, self._env.num_rows, self._env.num_cols, 3),
                dtype=bool,
                name="agents_view",
                minimum=0,
                maximum=self.num_agents,
            )
            obs_data["global_state"] = global_state
            return specs.Spec(ObservationGlobalState, "ObservationSpec", **obs_data)

        return specs.Spec(Observation, "ObservationSpec", **obs_data)
