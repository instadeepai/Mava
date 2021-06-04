# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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

from typing import Any, Dict, List

import dm_env
import numpy as np
from acme import specs
from acme.wrappers.open_spiel_wrapper import OpenSpielWrapper, rl_environment

from mava import types
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
from mava.wrappers.env_wrappers import SequentialEnvWrapper


class SequentialOpenSpielWrapper(SequentialEnvWrapper):
    def __init__(
        self,
        environment: rl_environment.Environment,
    ):
        acme_open_spiel_wrapper = OpenSpielWrapper(environment)
        self._environment = acme_open_spiel_wrapper
        self._possible_agents = [
            f"agent_{i}" for i in range(self._environment.num_players)
        ]
        self._agents = self._possible_agents[:]

    def reset(self) -> dm_env.TimeStep:
        """Resets the episode."""
        tm_step = self._environment.reset()
        observations = tm_step.observations
        rewards = {
            agent: convert_np_type(self.reward_spec()[agent].dtype, 0)
            for agent in self.possible_agents
        }
        self._discounts = {
            agent: convert_np_type(self.discount_spec()[agent].dtype, 1)
            for agent in self.possible_agents
        }
        return parameterized_restart(rewards, self._discounts, observations)

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps the environment."""
        agent = self.current_agent
        action = actions[agent]
        tm_step = self._environment.step(action)
        observations = {
            f"agent_{k}": observe for k, observe in enumerate(tm_step.observations)
        }
        rewards = {f"agent_{k}": rew for k, rew in enumerate(tm_step.rewards)}
        discounts = {
            f"agent_{k}": discount for k, discount in enumerate(tm_step.discounts)
        }
        step_type = tm_step.step_type

        return dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=discounts,
            step_type=step_type,
        )

    def observation_spec(self) -> types.Observation:
        olt = self.acme_open_spiel_wrapper.observations_spec
        observation_specs = {}
        for agent in self.possible_agents:
            observation_specs[agent] = types.OLT(
                observation=olt.observation,
                legal_actions=olt.legat_actions,
                terminal=olt.terminal,
            )
        return observation_specs

    def action_spec(self) -> Dict[str, specs.DiscreteArray]:
        action_specs = {}
        action_spc = self.acme_open_spiel_wrapper.action_spec()
        for agent in self.possible_agents:
            action_specs[agent] = action_spc
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        reward_specs = {}
        reward_spc = self.acme_open_spiel_wrapper.reward_spec()
        for agent in self.possible_agents:
            reward_specs[agent] = reward_spc

        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        discount_specs = {}
        discount_spc = self.acme_open_spiel_wrapper.discount_spec()
        for agent in self._environment.possible_agents:
            discount_specs[agent] = discount_spc
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        return {}

    @property
    def agents(self) -> List:
        return self._agents

    @property
    def possible_agents(self) -> List:
        return self._possible_agents

    @property
    def environment(self) -> rl_environment.Environment:
        """Returns the wrapped environment."""
        return self._environment

    @property
    def current_agent(self) -> Any:
        return f"agent_{self._environment.get_state.current_player()}"

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        return getattr(self._environment, name)
