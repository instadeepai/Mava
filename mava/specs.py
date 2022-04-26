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

from types import SimpleNamespace
from typing import Any, Dict, List

import dm_env
from acme.specs import EnvironmentSpec

from mava.utils.id_utils import EntityId
from mava.utils.sort_utils import sort_str_num


class MAEnvironmentSpec:
    def __init__(
        self,
        environment: dm_env.Environment,
        specs: Dict[str, EnvironmentSpec] = None,
        extra_specs: Dict = None,
    ):
        """_summary_

        Args:
            environment : _description_
            specs : _description_.
            extra_specs : _description_.
        """
        if not specs:
            specs = self._make_ma_environment_spec(environment)
        else:
            self.extra_specs = extra_specs
        self._keys = sort_str_num(list(map(str, specs.keys())))
        self._specs = {key: specs[key] for key in self._keys}

    def _make_ma_environment_spec(
        self, environment: dm_env.Environment
    ) -> Dict[str, EnvironmentSpec]:
        """_summary_

        Args:
            environment : _description_
        Returns:
            _description_
        """
        specs = {}
        observation_specs = environment.observation_spec()
        action_specs = environment.action_spec()
        reward_specs = environment.reward_spec()
        discount_specs = environment.discount_spec()
        self.extra_specs = environment.extra_spec()
        for agent in environment.possible_agents:
            agent = str(agent)
            specs[agent] = EnvironmentSpec(
                observations=observation_specs[agent],
                actions=action_specs[agent],
                rewards=reward_specs[agent],
                discounts=discount_specs[agent],
            )
        return specs

    def get_extra_specs(self) -> Dict[str, EnvironmentSpec]:
        """_summary_

        Returns:
            _description_
        """
        return self.extra_specs  # type: ignore

    def get_agent_specs(self) -> Dict[str, EnvironmentSpec]:
        """_summary_

        Returns:
            _description_
        """
        return self._specs

    def get_agent_type_specs(self) -> Dict[str, EnvironmentSpec]:
        """_summary_

        Returns:
            _description_
        """
        specs = {}

        agent_types = list({agent.type for agent in self._keys})
        for agent_type in agent_types:
            agent_id = str(EntityId(id=0, type=agent_type))
            specs[agent_type] = self._specs[agent_id]
        return specs

    def get_agent_ids(self) -> List[str]:
        """_summary_

        Returns:
            _description_
        """
        return self._keys

    def get_agent_types(self) -> List[str]:
        """_summary_

        Returns:
            _description_
        """
        return list({EntityId.from_string(agent).type for agent in self._keys})

    def get_agents_by_type(self) -> Dict[str, List[str]]:
        """_summary_

        Returns:
            _description_
        """
        agents_by_type: Dict[str, List[str]] = {}
        agents_ids = self.get_agent_ids()
        agent_types = self.get_agent_types()
        for agent_type in agent_types:
            agents_by_type[agent_type] = []
            for agent in agents_ids:
                if agent_type in agent:
                    agents_by_type[agent_type].append(agent)
        return agents_by_type


class DesignSpec(SimpleNamespace):
    def __init__(self, **kwargs: Any) -> None:
        """_summary_"""
        super().__init__(**kwargs)

    def get(self) -> Dict[str, Any]:
        """_summary_

        Returns:
            _description_
        """
        return self.__dict__
