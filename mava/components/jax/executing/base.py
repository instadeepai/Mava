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
from typing import Callable, List, Optional, Type

from mava.callbacks import Callback
from mava.components.jax import Component
from mava.core_jax import SystemExecutor


@dataclass
class ExecutorInitConfig:
    interval: Optional[dict] = None


class ExecutorInit(Component):
    def __init__(self, config: ExecutorInitConfig = ExecutorInitConfig()):
        """Component for initialising store parameters required for executor components.

        Args:
            config: ExecutorInitConfig.
        """
        self.config = config

    def on_execution_init_start(self, executor: SystemExecutor) -> None:
        """Save the interval from the config to the executor.

        Args:
            executor: SystemExecutor.

        Returns:
            None.
        """
        executor.store.policy_states = {}
        executor.store._interval = self.config.interval  # type: ignore

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "executor_init"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return ExecutorInitConfig

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []
