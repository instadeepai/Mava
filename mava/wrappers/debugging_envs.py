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
from typing import Any, Dict, Union

import dm_env
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec

from mava.utils.debugging.environment import MultiAgentEnv
from mava.utils.wrapper_utils import OLT, convert_np_type, parameterized_restart


class DebuggingEnvWrapper(dm_env.Environment):
    """Environment wrapper for PettingZoo MARL environments."""

    # Note: we don't inherit from base.EnvironmentWrapper because that class
    # assumes that the wrapped environment is a dm_env.Environment.
    def __init__(self, environment: MultiAgentEnv, render: bool = False):
        self._environment = environment
        self._reset_next_step = True
        self.possible_agents = self._environment.agent_ids
        self.render = render

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST
        discount_spec = self.discount_spec()
        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self._environment.agent_ids
        }
        observe = self._environment.reset()

        observations = self._convert_observations(
            observe, {agent: False for agent in self.agent_ids}
        )
        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self.agent_ids
        }

        discount_spec = self.discount_spec()
        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self.agent_ids
        }
        return parameterized_restart(rewards, self._discounts, observations)

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps the environment."""

        if self._reset_next_step:
            return self.reset()

        observations, rewards, dones, infos = self._environment.step(actions)

        if self.render:
            self._environment.render(mode="not_human")
            import time

            time.sleep(0.1)

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

    # Convert PettingZoo observation so it's dm_env compatible. Also, the list
    # of legal actions must be converted to a legal actions mask.
    def _convert_observations(
        self, observes: Dict[str, np.ndarray], dones: Dict[str, bool]
    ) -> Dict[str, OLT]:
        observations: Dict[str, OLT] = {}
        for agent, observation in observes.items():
            if isinstance(observation, dict) and "action_mask" in observation:
                legals = observation["action_mask"]
                observation = observation["observation"]
            else:
                # TODO Handle legal actions better for continous envs,
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

    def action_spec(self) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        action_specs = {}
        action_spaces = self._environment.action_spaces
        for agent in self._environment.agent_ids:
            action_specs[agent] = _convert_to_spec(action_spaces[agent])
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        reward_specs = {}
        for agent in self._environment.agent_ids:
            reward_specs[agent] = specs.Array((), np.float32)

        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        discount_specs = {}
        for agent in self._environment.agent_ids:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        return {}

    def seed(self, seed: int = None) -> None:
        self._environment.seed(seed)

    @property
    def environment(self) -> MultiAgentEnv:
        """Returns the wrapped environment."""
        return self._environment

    @property
    def current_agent(self) -> Any:
        return self._environment.agent_selection

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        return getattr(self._environment, name)
