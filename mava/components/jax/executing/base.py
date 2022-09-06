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
# For Precommit
import copy
from dataclasses import dataclass
from typing import Callable, Optional

from mava.components.jax import Component
from mava.core_jax import SystemBuilder, SystemExecutor


@dataclass
class ExecutorInitConfig:
    interval: Optional[dict] = None


class ExecutorInit(Component):
    def __init__(self, config: ExecutorInitConfig = ExecutorInitConfig()):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_init(self, builder: SystemBuilder) -> None:
        """Summary"""
        builder.store.networks = builder.store.network_factory()

    def on_execution_init_start(self, executor: SystemExecutor) -> None:
        """_summary_"""
        executor._interval = self.config.interval  # type: ignore

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "executor_init"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return ExecutorInitConfig


@dataclass
class ExecutorTargetNetInitConfig:
    pass


class ExecutorTargetNetInit(Component):
    def __init__(
        self, config: ExecutorTargetNetInitConfig = ExecutorTargetNetInitConfig()
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_init(self, builder: SystemBuilder) -> None:
        """Summary"""
        # Setup agent target networks
        builder.store.target_networks = copy.deepcopy(builder.store.networks)

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "executor_target_network_init"

    @staticmethod
    def config_class() -> Callable:
        """Returns the config class for this component."""
        return ExecutorTargetNetInitConfig
