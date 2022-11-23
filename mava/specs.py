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

from mava.utils.sort_utils import sort_str_num


class MAEnvironmentSpec:
    def __init__(
        self,
        environment: dm_env.Environment,
        agent_environment_specs: Dict[str, EnvironmentSpec] = None,
        extras_specs: Dict[str, Any] = None,
    ):
        """Multi-agent environment spec

        Create a multi-agent environment spec through a dm_env Environment
        or through pre-existing enviromnent specs (specifying an environment
        spec for each agent) and extras specs

        Args:
            environment : dm_env.Environment object
            agent_environment_specs : environment specs for each agent
            extras_specs : extras specs for additional data not contained
            in the acme EnvironmentSpec format, such as global state information
        """
        if not agent_environment_specs:
            agent_environment_specs = self._make_ma_environment_spec(environment)
        else:
            self._extras_specs = extras_specs
        self._keys = list(sort_str_num(agent_environment_specs.keys()))
        self._agent_environment_specs = {
            key: agent_environment_specs[key] for key in self._keys
        }

    def _make_ma_environment_spec(
        self, environment: dm_env.Environment
    ) -> Dict[str, EnvironmentSpec]:
        """Create a multi-agent environment spec from a dm_env environment

        Args:
            environment : dm_env.Environment
        Returns:
            Dictionary with an environment spec for each agent
        """
        agent_environment_specs = {}
        observation_specs = environment.observation_spec()
        action_specs = environment.action_spec()
        reward_specs = environment.reward_spec()
        valid_step_specs = environment.valid_step_spec()
        self._extras_specs = environment.extras_spec()
        for agent in environment.possible_agents:
            agent_environment_specs[agent] = EnvironmentSpec(
                observations=observation_specs[agent],
                actions=action_specs[agent],
                rewards=reward_specs[agent],
                valid_steps=valid_step_specs[agent],
            )
        return agent_environment_specs

    def get_extras_specs(self) -> Dict[str, Any]:
        """Get extras specs

        Returns:
            Extras spec that contains any additional information not contained
            within the environment specs
        """
        return self._extras_specs  # type: ignore

    def get_agent_environment_specs(self) -> Dict[str, EnvironmentSpec]:
        """Get environment specs for all agents

        Returns:
            Dictionary of environment specs, representing each agent in the environment
        """
        return self._agent_environment_specs

    def set_extras_specs(self, extras_specs: Dict[str, Any]) -> None:
        """Set extras specs

        Returns:
            None
        """
        self._extras_specs = extras_specs

    def set_agent_environment_specs(
        self, agent_environment_specs: Dict[str, EnvironmentSpec]
    ) -> None:
        """Set agent environment specs

        Returns:
            None
        """
        self._agent_environment_specs = agent_environment_specs

    def get_agent_type_specs(self) -> Dict[str, EnvironmentSpec]:
        """Get environment specs for all agent types

        Returns:
            Dictionary of environment specs, representing each agent type
            in the environment
        """
        agent_environment_specs = {}
        agent_types = list({agent.split("_")[0] for agent in self._keys})
        for agent_type in agent_types:
            agent_environment_specs[agent_type] = self._agent_environment_specs[
                f"{agent_type}_0"
            ]
        return agent_environment_specs

    def get_agent_ids(self) -> List[str]:
        """Get agent ids

        Returns:
            List of agent ids
        """
        return self._keys

    def get_agent_types(self) -> List[str]:
        """Get agent types

        Returns:
            List of agent types as defined by the ids of each agent
        """
        return list({agent.split("_")[0] for agent in self._keys})

    def get_agents_by_type(self) -> Dict[str, List[str]]:
        """Get agents by type

        Returns:
            Dictionary representing agents that belong to each agent type
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
        """Multi-agent system design spec

        This design spec is used to store a reference to all the components
        that should be included in a system.

        Args:
            uninitialised components.
        """
        super().__init__(**kwargs)

    def get(self) -> Dict[str, Any]:
        """Get the design spec dictionary.

        Returns:
            The dictionary inside the design spec.
        """
        return self.__dict__

    def set(self, name: str, component: Any) -> None:
        """Set/override a component in the design spec dictionary.

        Returns:
            None
        """
        self.__dict__[name] = component

    def set(self, component_dict: Dict) -> None:  # type: ignore
        """Set/override multiple components in the design spec dictionary.

        Returns:
            None
        """
        for key in component_dict.keys():
            self.__dict__[key] = component_dict[key]
