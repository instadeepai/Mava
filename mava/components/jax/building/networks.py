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

import abc
from dataclasses import dataclass
from typing import Callable, Optional

import dm_env
import jax

from mava.components.jax import Component
from mava.core_jax import SystemBuilder


@dataclass
class NetworksConfig:
    network_factory: Optional[Callable[[str], dm_env.Environment]] = None
    seed: int = 1234


class Networks(Component):
    @abc.abstractmethod
    def __init__(
        self,
        config: NetworksConfig = NetworksConfig(),
    ):
        """[summary]"""
        self.config = config

    @abc.abstractmethod
    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Summary"""
        pass

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "networks"


class DefaultNetworks(Networks):
    def __init__(
        self,
        config: NetworksConfig = NetworksConfig(),
    ):
        """[summary]"""
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Summary"""
        # Setup the jax key for network initialisations
        builder.store.key = jax.random.PRNGKey(self.config.seed)

        # Build network function here
        network_key, builder.store.key = jax.random.split(builder.store.key)
        builder.store.network_factory = lambda: self.config.network_factory(
            environment_spec=builder.store.agent_environment_specs,
            agent_net_keys=builder.store.agent_net_keys,
            rng_key=network_key,
        )

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return NetworksConfig
