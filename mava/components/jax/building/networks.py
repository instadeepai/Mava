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
from typing import Callable, List, Optional, Type

import dm_env
import jax

from mava.callbacks import Callback
from mava.components.jax import Component
from mava.components.jax.building.environments import EnvironmentSpec
from mava.components.jax.building.system_init import BaseSystemInit
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
        """Abstract component defining the skeleton for initialising networks.

        Args:
            config: NetworksConfig.
        """
        self.config = config

    @abc.abstractmethod
    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Create and store the network factory from the config.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        pass

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "networks"


class DefaultNetworks(Networks):
    def __init__(
        self,
        config: NetworksConfig = NetworksConfig(),
    ):
        """Component defines the default way to initialise networks.

        Args:
            config: NetworksConfig.
        """
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Create and store the network factory from the config.

        Also manages keys, creating and storing a key from the config seed.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        # Setup the jax key for network initialisations
        builder.store.base_key = jax.random.PRNGKey(self.config.seed)

        # Build network function here
        network_key, builder.store.base_key = jax.random.split(builder.store.base_key)
        builder.store.network_factory = lambda: self.config.network_factory(
            environment_spec=builder.store.ma_environment_spec,
            agent_net_keys=builder.store.agent_net_keys,
            rng_key=network_key,
        )

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        EnvironmentSpec required to set up builder.store.environment_spec.
        BaseSystemInit required to set up builder.store.agent_net_keys.

        Returns:
            List of required component classes.
        """
        return [EnvironmentSpec, BaseSystemInit]
