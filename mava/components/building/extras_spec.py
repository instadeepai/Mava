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

"""
ExtrasSpec's are used on creation of reverb tables, to set the table's signature for
data that needs to go into the table, but is not included in the environment spec's 
observations, actions or rewards
"""

import abc
from typing import Any, List

from dm_env import specs

from mava.components import Component


class ExtrasSpec(Component):
    @abc.abstractmethod
    def __init__(self, config: Any) -> None:
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

    def get_network_keys(self, unique_net_keys: List[str], agent_ids: List[str]):
        """
        Generates the network keys from the unique_net_keys and agent_ids, which
         is used by the reverb adders to place experience in the correct tables

        Params:
            unique_net_keys: keys of all the networks
            agent_ids: the IDs of all agents

        Returns:
            A dictionary of network keys mapping agents to their networks
        """
        int_spec = specs.DiscreteArray(len(unique_net_keys))
        return {"network_keys": {agent_id: int_spec for agent_id in agent_ids}}
