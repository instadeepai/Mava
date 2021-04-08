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

from typing import Any, Dict, List, Sequence, Union

import dm_env
import numpy as np
from acme import specs, types
from acme.testing.fakes import Actor as ActorMock
from acme.testing.fakes import _generate_from_spec, _validate_spec

from mava import core
from mava import specs as mava_specs
from mava.systems.system import System

"""Mock Objects for Tests"""


class MockedExecutor(ActorMock, core.Executor):
    """ Mock Exexutor Class."""

    def __init__(self, spec: specs.EnvironmentSpec):
        super().__init__(spec)

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Dict[str, types.NestedArray]:
        return {
            agent: _generate_from_spec(self._spec[agent].actions)
            for agent, observation in observations.items()
        }

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> Union[float, int]:
        return _generate_from_spec(self._spec[agent].actions)

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        for agent, observation_spec in self._spec.items():
            _validate_spec(observation_spec.observations, timestep.observation[agent])

    def agent_observe_first(self, agent: str, timestep: dm_env.TimeStep) -> None:
        _validate_spec(self._spec[agent].observations, timestep.observation)

    def observe(
        self,
        action: types.NestedArray,
        next_timestep: dm_env.TimeStep,
    ) -> None:

        for agent, observation_spec in self._spec.items():
            _validate_spec(observation_spec.actions, action[agent])
            _validate_spec(observation_spec.rewards, next_timestep.reward[agent])
            _validate_spec(observation_spec.discounts, next_timestep.discount[agent])
            if next_timestep.observation:
                _validate_spec(
                    observation_spec.observations, next_timestep.observation[agent]
                )

    def agent_observe(
        self,
        agent: str,
        action: Union[float, int],
        next_timestep: dm_env.TimeStep,
    ) -> None:
        observation_spec = self._spec[agent]
        _validate_spec(observation_spec.actions, action)
        _validate_spec(observation_spec.rewards, next_timestep.reward)
        _validate_spec(observation_spec.discounts, next_timestep.discount)
        _validate_spec(observation_spec.observations, next_timestep.observation)


class MockedSystem(MockedExecutor, System):
    """Mocked System Class. """

    def __init__(
        self,
        spec: specs.EnvironmentSpec,
    ):
        super().__init__(spec)

        # Ini Mock Vars
        self.variables: Dict = {}
        network_type = "mlp"
        self.variables[network_type] = {}
        for agent in self._spec.keys():
            self.variables[network_type][agent] = np.random.rand(5, 5)

    def get_variables(self, names: Dict[str, Sequence[str]]) -> Dict[str, List[Any]]:
        variables: Dict = {}
        for network_type in names:
            variables[network_type] = {}
            for agent in self.agents:
                variables[network_type][agent] = self.variables[network_type][agent]
        return variables


def get_mocked_env_spec(environment: specs.EnvironmentSpec) -> dm_env.Environment:
    return mava_specs.MAEnvironmentSpec(environment)
