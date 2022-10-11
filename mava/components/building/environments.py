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
import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple, Type, Union

import acme

from mava import specs
from mava.callbacks import Callback
from mava.components import Component
from mava.components.building.loggers import Logger
from mava.core_jax import SystemBuilder
from mava.environment_loop import ParallelEnvironmentLoop
from mava.utils.sort_utils import sort_str_num
from mava.wrappers.environment_loop_wrappers import (
    DetailedPerAgentStatistics,
    EnvironmentLoopStatisticsBase,
    MonitorParallelEnvironmentLoop,
)


@dataclass
class EnvironmentSpecConfig:
    environment_factory: Optional[Callable[[bool], acme.core.Worker]] = None


class EnvironmentSpec(Component):
    def __init__(self, config: EnvironmentSpecConfig = EnvironmentSpecConfig()):
        """Component creates a multi-agent environment spec.

        Args:
            config: EnvironmentSpecConfig.
        """
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Using the env factory in config, create and store the env spec and agents.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        builder.store.manager_pid = os.getpid()
        builder.store.ma_environment_spec = specs.MAEnvironmentSpec(
            self.config.environment_factory()
        )

        builder.store.agents = sort_str_num(
            builder.store.ma_environment_spec.get_agent_ids()
        )
        builder.store.extras_spec = {}

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "environment_spec"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []


@dataclass
class ExecutorEnvironmentLoopConfig:
    should_update: bool = True
    executor_stats_wrapper_class: Optional[
        Type[EnvironmentLoopStatisticsBase]
    ] = DetailedPerAgentStatistics
    sum_episode_return: bool = False


class ExecutorEnvironmentLoop(Component):
    def __init__(
        self, config: ExecutorEnvironmentLoopConfig = ExecutorEnvironmentLoopConfig()
    ):
        """Component creates an executor environment loop.

        Args:
            config: ExecutorEnvironmentLoopConfig.
        """
        self.config = config

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """Create and store the executor environment from the factory in config.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        # Global config set by EnvironmentSpec component
        builder.store.executor_environment = (
            builder.store.global_config.environment_factory(evaluation=False)
        )  # type: ignore

    @abc.abstractmethod
    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """Abstract method for overriding: should create executor environment loop.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "executor_environment_loop"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        Logger required to set up builder.store.executor_logger.
        EnvironmentSpec required for config environment_factory.

        Returns:
            List of required component classes.
        """
        return [Logger, EnvironmentSpec]


class ParallelExecutorEnvironmentLoop(ExecutorEnvironmentLoop):
    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """Create and store a parallel environment loop.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        executor_environment_loop = ParallelEnvironmentLoop(
            environment=builder.store.executor_environment,
            executor=builder.store.executor,  # Set up by builder
            logger=builder.store.executor_logger,
            should_update=self.config.should_update,
            sum_episode_return=self.config.sum_episode_return,
        )
        del builder.store.executor_logger

        if self.config.executor_stats_wrapper_class:
            executor_environment_loop = self.config.executor_stats_wrapper_class(
                executor_environment_loop
            )
        builder.store.system_executor = executor_environment_loop


@dataclass
class MonitorExecutorEnvironmentLoopConfig(ExecutorEnvironmentLoopConfig):
    filename: str = "agents"
    label: str = "parallel_environment_loop"
    record_every: int = 1000
    fps: int = 15
    counter_str: str = "evaluator_episodes"
    format: str = "video"
    figsize: Union[float, Tuple[int, int]] = (360, 640)


class MonitorExecutorEnvironmentLoop(ExecutorEnvironmentLoop):
    def __init__(
        self,
        config: MonitorExecutorEnvironmentLoopConfig = MonitorExecutorEnvironmentLoopConfig(),  # noqa
    ):
        """Component for visualising environment progress."""
        super().__init__(config=config)
        self.config = config

    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """Monitors environments and produces videos of episodes.

        Builds a `ParallelEnvironmentLoop` on the evaluator and a
        `MonitorParallelEnvironmentLoop` on all executors and stores it
        in the `builder.store.system_executor`.

        Args:
            builder: SystemBuilder
        """
        if builder.store.is_evaluator:
            executor_environment_loop = MonitorParallelEnvironmentLoop(
                environment=builder.store.executor_environment,
                executor=builder.store.executor,
                logger=builder.store.executor_logger,
                should_update=self.config.should_update,
                filename=self.config.filename,
                label=self.config.label,
                record_every=self.config.record_every,
                path=builder.store.global_config.experiment_path,
                fps=self.config.fps,
                counter_str=self.config.counter_str,
                format=self.config.format,
                figsize=self.config.figsize,
            )
        else:
            executor_environment_loop = ParallelEnvironmentLoop(
                environment=builder.store.executor_environment,
                executor=builder.store.executor,
                logger=builder.store.executor_logger,
                should_update=self.config.should_update,
            )

        del builder.store.executor_logger

        if self.config.executor_stats_wrapper_class:
            executor_environment_loop = self.config.executor_stats_wrapper_class(
                executor_environment_loop
            )

        builder.store.system_executor = executor_environment_loop
