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

"""Wraps a Debugging MARL environment to be used as a dm_env environment."""
from typing import Dict, Tuple

import dm_env
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from gym import spaces

from mava.types import OLT
from mava.utils.debugging.environment import MultiAgentEnv
from mava.utils.debugging.environments.switch_game import MultiAgentSwitchGame
from mava.utils.debugging.environments.two_step import TwoStepEnv
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
from mava.wrappers.pettingzoo import PettingZooParallelEnvWrapper


class DebuggingEnvWrapper(PettingZooParallelEnvWrapper):
    """Environment wrapper for Debugging MARL environments."""

    def __init__(
        self,
        environment: MultiAgentEnv,
        return_state_info: bool = False,
    ):
        super().__init__(environment=environment)

        self.return_state_info = return_state_info

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST
        discount_spec = self.discount_spec()
        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self._environment.possible_agents
        }
        observe, env_extras = self._environment.reset()

        observations = self._convert_observations(
            observe, {agent: False for agent in self.possible_agents}
        )
        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self.possible_agents
        }
        if not self.return_state_info:
            env_extras = {}

        return parameterized_restart(rewards, self._discounts, observations), env_extras

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[dm_env.TimeStep, np.array]:
        """Steps the environment."""

        if self._reset_next_step:
            return self.reset()

        observations, rewards, dones, state = self._environment.step(actions)

        rewards_spec = self.reward_spec()
        #  Handle empty rewards
        if not rewards:
            rewards = {
                agent: convert_np_type(rewards_spec[agent].dtype, 0)
                for agent in self.agent_ids
            }
        else:
            rewards = {
                agent: convert_np_type(rewards_spec[agent].dtype, reward)
                for agent, reward in rewards.items()
            }

        if observations:
            observations = self._convert_observations(observations, dones)

        if self._environment.env_done:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
        else:
            self._step_type = dm_env.StepType.MID

        timestep = dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=self._discounts,
            step_type=self._step_type,
        )
        if self.return_state_info:
            return timestep, {"s_t": state}
        else:
            return timestep

    # Convert Debugging environment observation so it's dm_env compatible.
    # Also, the list of legal actions must be converted to a legal actions mask.
    def _convert_observations(
        self, observes: Dict[str, np.ndarray], dones: Dict[str, bool]
    ) -> Dict[str, OLT]:
        observations: Dict[str, OLT] = {}
        for agent, observation in observes.items():
            if isinstance(observation, dict) and "action_mask" in observation:
                legals = observation["action_mask"]
                observation = observation["observation"]
            else:
                # TODO Handle legal actions better for continuous envs,
                #  maybe have min and max for each action and clip the agents actions
                #  accordingly
                legals = np.ones(
                    _convert_to_spec(self._environment.action_spaces[agent]).shape,
                    dtype=self._environment.action_spaces[agent].dtype,
                )

            observation = np.array(observation, dtype=np.float32)
            observations[agent] = OLT(
                observation=observation,
                legal_actions=legals,
                terminal=np.asarray([dones[agent]], dtype=np.float32),
            )

        return observations

    def observation_spec(self) -> Dict[str, OLT]:
        observation_specs = {}
        for agent in self._environment.agent_ids:
            observation_specs[agent] = OLT(
                observation=_convert_to_spec(
                    self._environment.observation_spaces[agent]
                ),
                legal_actions=_convert_to_spec(self._environment.action_spaces[agent]),
                terminal=specs.Array((1,), np.float32),
            )
        return observation_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        extras = {}
        if self.return_state_info:
            shape = self.environment._get_state().shape

            ex_spec = specs.BoundedArray(
                shape=shape,
                dtype="float32",
                name="observation",
                minimum=[float("-inf")] * shape[0],
                maximum=[float("inf")] * shape[0],
            )
            extras.update({"s_t": ex_spec})
        return extras


class SwitchGameWrapper(PettingZooParallelEnvWrapper):
    """Environment wrapper for Debugging Switch environment."""

    def __init__(self, environment: MultiAgentSwitchGame):
        super().__init__(environment=environment)

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        if self._reset_next_step:
            return self.reset()

        observations, rewards, dones, infos = self._environment.step(actions)

        if observations:
            observations = self._convert_observations(observations, dones)

        if self._environment.env_done:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
        else:
            self._step_type = dm_env.StepType.MID

        return dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=self._discounts,
            step_type=self._step_type,
        )

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        return {}


class TwoStepWrapper(PettingZooParallelEnvWrapper):
    """Wraps simple two-step matrix game from Qmix paper. Useful for
    debugging and quick comparison of cooperative performance."""

    def __init__(self, environment: TwoStepEnv) -> None:
        super().__init__(environment=environment)
        self._reset_next_step = False
        self.environment.action_spaces = {}
        self.environment.observation_spaces = {}
        self.environment.extra_specs = {
            "s_t": spaces.Discrete(3)
        }  # Global state 1, 2, or 3

        for agent_id in self.environment.agent_ids:
            self.environment.action_spaces[agent_id] = spaces.Discrete(2)  # int64
            self.environment.observation_spaces[agent_id] = spaces.Box(
                0, 1, shape=(1,)
            )  # float32

        self.reset()

    def step(self, actions: Dict[str, np.array]) -> Tuple[dm_env.TimeStep, np.array]:
        """Steps the environment."""
        if self._reset_next_step:
            self._reset_next_step = False
            self.reset()

        observations, rewards, dones, state_infos = self._environment.step(actions)
        if observations:
            observations = self._convert_observations(observations, dones)

        if self._environment.env_done:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
        else:
            self._step_type = dm_env.StepType.MID

        return (
            dm_env.TimeStep(
                observation=observations,
                reward=rewards,
                discount=self._discounts,
                step_type=self._step_type,
            ),
            state_infos,
        )

    def reset(self) -> Tuple[dm_env.TimeStep, np.array]:
        """Resets the episode."""
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST
        discount_spec = self.discount_spec()
        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self._environment.possible_agents
        }
        observe, state_infos = self._environment.reset()
        observations = self._convert_observations(
            observe, {agent: False for agent in self.possible_agents}
        )
        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self.possible_agents
        }

        discount_spec = self.discount_spec()
        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self.possible_agents
        }
        return (
            parameterized_restart(rewards, self._discounts, observations),
            state_infos,
        )

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        return self.environment.extra_specs
