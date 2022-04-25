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

import acme

from mava import specs
from mava.components.jax import Component
from mava.core_jax import SystemBuilder
from mava.environment_loop import JAXParallelEnvironmentLoop, ParallelEnvironmentLoop


@dataclass
class ExecutorEnvironmentLoopConfig:
    environment_factory: Optional[Callable[[bool], acme.core.Worker]] = None
    should_update: bool = True


class ExecutorEnvironmentLoop(Component):
    def __init__(
        self, config: ExecutorEnvironmentLoopConfig = ExecutorEnvironmentLoopConfig()
    ):
        """[summary]"""
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        builder.store.environment_spec = specs.MAEnvironmentSpec(
            self.config.environment_factory(evaluation=False)  # type: ignore
        )

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        builder.store.executor_environment = self.config.environment_factory(
            evaluation=False
        )  # type: ignore
        builder.store.environment_spec = specs.MAEnvironmentSpec(
            builder.store.executor_environment
        )

    @abc.abstractmethod
    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "environment_loop"

    @staticmethod
    def config_class() -> Callable:
        return ExecutorEnvironmentLoopConfig


class ParallelExecutorEnvironmentLoop(ExecutorEnvironmentLoop):
    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        executor_environment_loop = ParallelEnvironmentLoop(
            environment=builder.store.executor_environment,
            executor=builder.store.executor,
            logger=builder.store.executor_logger,
            should_update=self.config.should_update,
        )
        del builder.store.executor_logger

        builder.store.system_executor = executor_environment_loop


@dataclass
class JAXParallelExecutorEnvironmentLoopConfig(ExecutorEnvironmentLoopConfig):
    rng_seed: int = 0


class JAXParallelExecutorEnvironmentLoop(ExecutorEnvironmentLoop):
    def __init__(
        self,
        config: JAXParallelExecutorEnvironmentLoopConfig = JAXParallelExecutorEnvironmentLoopConfig(),
    ):
        self.config = config

    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        executor_environment_loop = JAXParallelEnvironmentLoop(
            environment=builder.store.executor_environment,
            executor=builder.store.executor,
            logger=builder.store.executor_logger,
            should_update=self.config.should_update,
            rng_seed=self.config.rng_seed,
        )
        del builder.store.executor_logger

        builder.store.system_executor = executor_environment_loop

    @staticmethod
    def config_class() -> Callable:
        return JAXParallelExecutorEnvironmentLoopConfig
