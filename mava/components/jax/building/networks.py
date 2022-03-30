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

"""Execution components for system builders"""

from dataclasses import dataclass
from typing import Callable, Optional

import dm_env

from mava.components.jax import Component
from mava.core_jax import SystemBuilder


@dataclass
class NetworksConfig:
    network_factory: Optional[Callable[[str], dm_env.Environment]] = None
    shared_weights: bool = True


class DefaultNetworks(Component):
    def __init__(
        self,
        config: NetworksConfig = NetworksConfig(),
    ):
        """[summary]"""
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Summary"""
        builder.attr.network_factory = self.config.network_factory
        builder.attr.shared_networks = self.config.shared_weights

    # def on_building_executor_start(self, builder: SystemBuilder) -> None:
    #     """_summary_"""
    #     builder.attr.networks = builder.attr.network_factory(
    #         environment_spec=builder.attr.environment_spec,
    #         agent_net_keys=builder.attr.agent_net_keys,
    #         net_spec_keys=builder.attr.net_spec_keys,
    #     ),

    @property
    def name(self) -> str:
        """_summary_"""
        return "networks"
