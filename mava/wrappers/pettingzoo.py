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

"""Wraps a PettingZoo MARL environment to be used as a dm_env environment."""
from typing import Any, Dict, NamedTuple, Union

import dm_env
import numpy as np
from acme import specs, types
from pettingzoo.utils.env import AECEnv
from pettingzoo.utils.env import ParallelEnv


class OLT(NamedTuple):
    """Container for (observation, legal_actions, terminal) tuples."""

    observation: types.Nest
    legal_actions: types.Nest
    terminal: types.Nest


class PettingZooAECEnvWrapper(dm_env.Environment):
    """Environment wrapper for PettingZoo MARL environments."""

    # Note: we don't inherit from base.EnvironmentWrapper because that class
    # assumes that the wrapped environment is a dm_env.Environment.
    def __init__(self, environment: AECEnv):
        self._environment = environment
        self._reset_next_step = True
        self._step_type = dm_env.StepType.FIRST

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        self._environment.reset()
        self._discount = None  # Not used in pettingzoo

        observe, _, done, _ = self._environment.last()
        agent = self._environment.agent_selection
        observation = self._convert_observation(agent, observe, done)
        return dm_env.restart(observation)

    def step(self, action: int) -> dm_env.TimeStep:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        observe, reward, done, info = self._environment.last()
        agent = self._environment.agent_selection
        if self._environment.dones[agent]:
            self._environment.step(None)
        else:
            self._environment.step(action)

        observation = self._convert_observation(agent, observe, done)

        if self._step_type == dm_env.StepType.FIRST:
            step_type = self._step_type
            self._step_type = dm_env.StepType.MID
        else:
            step_type = (
                dm_env.StepType.LAST
                if self._environment.env_done
                else dm_env.StepType.MID
            )
        self._reset_next_step = step_type == dm_env.StepType.LAST

        return dm_env.TimeStep(
            observation=observation,
            reward=reward,
            discount=self._discount,
            step_type=step_type,
        )

    # Convert PettingZoo observation so it's dm_env compatible. Also, the list
    # of legal actions must be converted to a legal actions mask.
    def _convert_observation(
        self, agent: str, observe: Union[dict, np.ndarray], done: bool
    ) -> OLT:
        if isinstance(observe, dict) and "action_mask" in observe:
            observe = observe["observation"]
            legals = observe["action_mask"]
        else:
            legals = np.ones(self._environment.action_spaces[agent].n, dtype=np.float32)
        observation = OLT(
            observation=observe,
            legal_actions=legals,
            terminal=np.asarray([done], dtype=np.float32),
        )
        return observation

    def observation_spec(self) -> Dict[str, OLT]:
        observation_specs = {}
        for agent in self._environment.possible_agents:
            observation_specs[agent] = OLT(
                observation=specs.Array(
                    self._environment.observation_spaces[agent].shape, np.float32
                ),
                legal_actions=specs.Array(
                    (self._environment.action_spaces[agent].n,), np.float32
                ),
                terminal=specs.Array((1,), np.float32),
            )
        return observation_specs

    def action_spec(self) -> Dict[str, specs.DiscreteArray]:
        action_specs = {}
        for agent in self._environment.possible_agents:
            action_specs[agent] = specs.DiscreteArray(
                self._environment.action_spaces[agent].n
            )
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        reward_specs = {}
        for agent in self._environment.possible_agents:
            reward_specs[agent] = specs.Array((), np.float32)
        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        discount_specs = {}
        for agent in self._environment.possible_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    @property
    def environment(self) -> AECEnv:
        """Returns the wrapped environment."""
        return self._environment

    @property
    def current_agent(self) -> Any:
        return self._environment.agent_selection

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        return getattr(self._environment, name)


class PettingZooParallelEnvWrapper(dm_env.Environment):
    """Environment wrapper for PettingZoo MARL environments."""

    # Note: we don't inherit from base.EnvironmentWrapper because that class
    # assumes that the wrapped environment is a dm_env.Environment.
    def __init__(self, environment: ParallelEnv):
        self._environment = environment
        self._reset_next_step = True
        self._step_type = dm_env.StepType.FIRST

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        self._discount = None  # Not used in pettingzoo
        observe = self._environment.reset()
        observations = self._convert_observations(observations)
        return dm_env.restart(observations)

    def step(self, actions: int) -> dm_env.TimeStep:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        observations, rewards, dones, infos = self._environment.step(actions)

        if self._environment.done:
            self._environment.step(None)
        else:
            self._environment.step(actions)

        observations = self._convert_observations(observations)

        if self._step_type == dm_env.StepType.FIRST:
            step_type = self._step_type
            self._step_type = dm_env.StepType.MID
        else:
            step_type = (
                dm_env.StepType.LAST
                if self._environment.env_done
                else dm_env.StepType.MID
            )
        self._reset_next_step = step_type == dm_env.StepType.LAST

        return dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=self._discount,
            step_type=step_type,
        )

    # Convert PettingZoo observation so it's dm_env compatible. Also, the list
    # of legal actions must be converted to a legal actions mask.
    def _convert_observations(self, observes: Dict[str, np.ndarray]) -> Dict[str, OLT]:
        observations = {}
        for agent, observation in observes.items():
            if isinstance(observation, dict) and "action_mask" in observation:
                observation = observation["observation"]
                legals = observation["action_mask"]
            else:
                legals = np.ones(
                    self._environment.action_spaces[agent].n, dtype=np.float32
                )
            observation[agent] = OLT(
                observation=observation,
                legal_actions=legals,
                terminal=np.asarray([self._environment.dones[agent]], dtype=np.float32),
            )
        return observations

    def observation_spec(self) -> Dict[str, OLT]:
        observation_specs = {}
        for agent in self._environment.possible_agents:
            observation_specs[agent] = OLT(
                observation=specs.Array(
                    self._environment.observation_spaces[agent].shape, np.float32
                ),
                legal_actions=specs.Array(
                    (self._environment.action_spaces[agent].n,), np.float32
                ),
                terminal=specs.Array((1,), np.float32),
            )
        return observation_specs

    def action_spec(self) -> Dict[str, specs.DiscreteArray]:
        action_specs = {}
        for agent in self._environment.possible_agents:
            action_specs[agent] = specs.DiscreteArray(
                self._environment.action_spaces[agent].n
            )
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        reward_specs = {}
        for agent in self._environment.possible_agents:
            reward_specs[agent] = specs.Array((), np.float32)
        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        discount_specs = {}
        for agent in self._environment.possible_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    @property
    def environment(self) -> AECEnv:
        """Returns the wrapped environment."""
        return self._environment

    @property
    def current_agent(self) -> Any:
        return self._environment.agent_selection

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        return getattr(self._environment, name)