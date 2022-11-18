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

"""ExtrasSpecs are used to inform the reverb table about extra data.

They are used on creation of reverb tables, to set the table's signature for
data that needs to go into the table, but is not included in the environments
observations, actions or rewards
"""

import abc
from types import SimpleNamespace
from typing import Dict, List

from dm_env import specs

from mava.components import Component
from mava.core_jax import SystemBuilder


class ExtrasSpec(Component):
    def __init__(self, config: SimpleNamespace = SimpleNamespace()) -> None:
        """Initialise extra specs

        Args:
            config : ExtrasSpecConfig
        """
        self.config = config

    @staticmethod
    def name() -> str:
        """Returns name of ExtrasSpec class

        Returns:
            "extras_spec": name of ExtrasSpec class
        """
        return "extras_spec"

    def get_network_keys(
        self, unique_net_keys: List, agent_ids: List[str]
    ) -> Dict[str, Dict[str, int]]:
        """Generates the network keys used by adders.

        Uses the the unique_net_keys and agent_ids, to generate network keys.
        The keys inform the reverb adders in which tables to place the experience.

        Args:
            unique_net_keys: keys of all the networks
            agent_ids: the IDs of all agents

        Returns:
            A dictionary of network keys mapping agents to their networks
        """
        int_spec = specs.DiscreteArray(len(unique_net_keys))
        return {"network_keys": {agent_id: int_spec for agent_id in agent_ids}}

    @abc.abstractmethod
    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Create extra specs after builder has been initialised

        Args:
            builder: SystemBuilder

        Returns:
            None.
        """
        pass
