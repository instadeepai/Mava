# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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
from collections import deque
from typing import TYPE_CHECKING, Any, Dict

import dm_env
import numpy as np

from mava.types import OLT

try:
    import pettingzoo  # noqa: F401

    _has_petting_zoo = True
except ModuleNotFoundError:
    _has_petting_zoo = False

if _has_petting_zoo:

    # Prevent circular import issue.
    if TYPE_CHECKING:
        from mava.wrappers.pettingzoo import PettingZooParallelEnvWrapper  # noqa: F401

PettingZooEnv = "PettingZooParallelEnvWrapper"


class BasePreprocessWrapper(ABC):
    def __init__(self, environment: Any) -> None:
        """Initialise wrapper."""
        self._environment = environment

    @abstractmethod
    def reset(self) -> dm_env.TimeStep:
        """Reset the environment."""

    @abstractmethod
    def step(self, actions: Dict) -> dm_env.TimeStep:
        """Step the environment."""

    def observation_spec(self) -> Dict[str, OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        timestep = self.reset()
        if type(timestep) == tuple:
            timestep, _ = timestep

        observations = timestep.observation
        return observations

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)


class ConcatAgentIdToObservation(BasePreprocessWrapper):
    """Concat one-hot vector of agent ID to obs.

    We assume the environment has an ordered list
    self.possible_agents. We also assume the observations
    are vector based.
    """

    def __init__(self, environment: Any) -> None:
        """Initialise wrapper."""
        super().__init__(environment)

        self._num_agents = len(environment.possible_agents)

        # Check that observation of first agent is a vector
        if (
            len(
                list(self._environment.observation_spec().values())[0].observation.shape
            )
            > 1
        ):
            raise NotImplementedError(
                "Agent ID concatenation is only implemented for vector\
                    based observations."
            )

    def reset(self) -> dm_env.TimeStep:
        """Reset environment and concat agent ID."""
        timestep = self._environment.reset()
        if type(timestep) == tuple:
            timestep, env_extras = timestep
        else:
            env_extras = {}

        old_observations = timestep.observation

        new_observations = {}

        for agent_id, agent in enumerate(self._environment.possible_agents):
            agent_olt = old_observations[agent]

            agent_observation = agent_olt.observation
            agent_one_hot = np.zeros(self._num_agents, dtype=agent_observation.dtype)
            agent_one_hot[agent_id] = 1

            new_observations[agent] = OLT(
                observation=np.concatenate([agent_one_hot, agent_observation]),
                legal_actions=agent_olt.legal_actions,
                agent_mask=agent_olt.agent_mask,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            env_extras,
        )

    def step(self, actions: Dict) -> dm_env.TimeStep:
        """Step the environment and concat agent ID"""
        timestep = self._environment.step(actions)
        if type(timestep) == tuple:
            timestep, env_extras = timestep
        else:
            env_extras = {}

        old_observations = timestep.observation
        new_observations = {}
        for agent_id, agent in enumerate(self._environment.possible_agents):
            agent_olt = old_observations[agent]

            agent_observation = agent_olt.observation
            agent_one_hot = np.zeros(self._num_agents, dtype=agent_observation.dtype)
            agent_one_hot[agent_id] = 1

            new_observations[agent] = OLT(
                observation=np.concatenate([agent_one_hot, agent_observation]),
                legal_actions=agent_olt.legal_actions,
                agent_mask=agent_olt.agent_mask,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            env_extras,
        )

    @property
    def obs_normalisation_start_index(self) -> int:
        """Returns an interger to indicate which features should not be normalised"""

        old_value = self._environment.obs_normalisation_start_index
        num_values = self._num_agents

        return old_value + num_values


class ConcatPrevActionToObservation(BasePreprocessWrapper):
    """Concat one-hot vector of agent prev_action to obs.

    We assume the environment has discreet actions.

    TODO (Claude) support continuous actions.
    """

    def __init__(self, environment: Any):
        """Initialise wrapper."""
        super().__init__(environment)

    def reset(self) -> dm_env.TimeStep:
        """Reset the environment and add zero action."""
        timestep = self._environment.reset()
        if type(timestep) == tuple:
            timestep, env_extras = timestep
        else:
            env_extras = {}

        old_observations = timestep.observation
        action_spec = self._environment.action_spec()
        new_observations = {}
        # TODO double check this, because possible agents could shrink
        for agent in self._environment.possible_agents:
            agent_olt = old_observations[agent]
            agent_observation = agent_olt.observation
            agent_one_hot_action = np.zeros(
                action_spec[agent].num_values, dtype=np.float32
            )

            new_observations[agent] = OLT(
                observation=np.concatenate([agent_one_hot_action, agent_observation]),
                legal_actions=agent_olt.legal_actions,
                agent_mask=agent_olt.agent_mask,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            env_extras,
        )

    def step(self, actions: Dict) -> dm_env.TimeStep:
        """Step the environment and concat prev actions."""
        timestep = self._environment.step(actions)
        if type(timestep) == tuple:
            timestep, env_extras = timestep
        else:
            env_extras = {}

        old_observations = timestep.observation
        action_spec = self._environment.action_spec()
        new_observations = {}
        for agent in self._environment.possible_agents:
            agent_olt = old_observations[agent]
            agent_observation = agent_olt.observation
            agent_one_hot_action = np.zeros(
                action_spec[agent].num_values, dtype=np.float32
            )
            agent_one_hot_action[actions[agent]] = 1

            new_observations[agent] = OLT(
                observation=np.concatenate([agent_one_hot_action, agent_observation]),
                legal_actions=agent_olt.legal_actions,
                agent_mask=agent_olt.agent_mask,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            env_extras,
        )

    @property
    def obs_normalisation_start_index(self) -> int:
        """Returns an interger to indicate which features should not be normalised"""

        old_value = self._environment.obs_normalisation_start_index
        action_spec = self._environment.action_spec()
        agent = self._environment.possible_agents[0]
        num_values = action_spec[agent].num_values

        return old_value + num_values


class StackObservations(BasePreprocessWrapper):
    """Stack frames of observations together"""

    def __init__(self, environment: Any, num_frames: int = 1):
        """Initialise wrapper."""

        super().__init__(environment)

        # Check that this is the first wrapper that is called on the environment
        if isinstance(self._environment, BasePreprocessWrapper):
            raise ValueError(
                "Observation stacking should be the first custom\
                    wrapper we call in the environment factory."
            )

        self.frames: Dict[str, deque] = {
            agent: deque([], maxlen=num_frames)
            for agent in self._environment.possible_agents
        }
        self.num_frames = num_frames

    def reset(self) -> dm_env.TimeStep:
        """Reset the environment by setting the same observation to all the frames"""

        timestep = self._environment.reset()
        if type(timestep) == tuple:
            timestep, env_extras = timestep
        else:
            env_extras = {}

        new_observations: Any = {}

        old_observations = timestep.observation
        for agent in self._environment.possible_agents:
            agent_olt = old_observations[agent]
            agent_observation = agent_olt.observation

            self.frames[agent] = deque(
                [agent_observation for _ in range(self.num_frames)],
                maxlen=self.num_frames,
            )

            agent_stack_observation = np.array(self.frames[agent])
            agent_stack_observation = agent_stack_observation.reshape(
                (-1,) + agent_observation.shape[2:]
            )
            new_observations[agent] = OLT(
                observation=agent_stack_observation,
                legal_actions=agent_olt.legal_actions,
                agent_mask=agent_olt.agent_mask,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            env_extras,
        )

    def step(self, actions: Dict) -> dm_env.TimeStep:
        """Step the environment and stack previous frames"""
        timestep = self._environment.step(actions)
        if type(timestep) == tuple:
            timestep, env_extras = timestep
        else:
            env_extras = {}

        new_observations: Any = {}

        old_observations = timestep.observation
        for agent in self._environment.possible_agents:
            agent_olt = old_observations[agent]
            agent_observation = agent_olt.observation

            self.frames[agent].append(agent_observation)

            agent_stack_observation = np.array(self.frames[agent])
            agent_stack_observation = agent_stack_observation.reshape(
                (-1,) + agent_observation.shape[2:]
            )
            new_observations[agent] = OLT(
                observation=agent_stack_observation,
                legal_actions=agent_olt.legal_actions,
                agent_mask=agent_olt.agent_mask,
            )

        return (
            dm_env.TimeStep(
                timestep.step_type, timestep.reward, timestep.discount, new_observations
            ),
            env_extras,
        )
