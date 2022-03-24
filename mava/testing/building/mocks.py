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

"""Tests for config class for Jax-based Mava systems"""

from dataclasses import dataclass, field
from typing import List

from mava.callbacks import Callback
from mava.core_jax import SystemBuilder


# Mock components to feed to the builder
@dataclass
class MockAdderSignatureConfig:
    adder_signature_param: int = 1


class MockAdderSignature(Callback):
    def __init__(
        self,
        config: MockAdderSignatureConfig = MockAdderSignatureConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.adder_signature_fn = self.config.adder_signature_param

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "adder_signature"


@dataclass
class MockAdderConfig:
    adder_param: float = 2.7


class MockAdder(Callback):
    def __init__(self, config: MockAdderConfig = MockAdderConfig()) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.adder = self.config.adder_param

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "adder"


@dataclass
class MockDataServerConfig:
    data_server_param: int = 2


class MockDataServer(Callback):
    def __init__(
        self,
        config: MockDataServerConfig = MockDataServerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_data_server(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.system_data_server = self.config.data_server_param

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "data_server"


@dataclass
class MockParameterServerConfig:
    parameter_server_param: int = 2


class MockParameterServer(Callback):
    def __init__(
        self,
        config: MockParameterServerConfig = MockParameterServerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_parameter_server(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.system_parameter_server = self.config.parameter_server_param

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "parameter_server"


@dataclass
class MockLoggerConfig:
    logger_param_0: int = 1
    logger_param_1: int = 2


class MockLogger(Callback):
    def __init__(
        self,
        config: MockLoggerConfig = MockLoggerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.executor_logger = self.config.logger_param_0

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.trainer_logger = self.config.logger_param_0

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "logger"


@dataclass
class MockParameterClientConfig:
    parameter_client_param_0: int = 1
    parameter_client_param_1: str = "param"


class MockParameterClient(Callback):
    def __init__(
        self,
        config: MockParameterClientConfig = MockParameterClientConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.executor_parameter_client = (
            builder._parameter_server_client,
            self.config.parameter_client_param_0,
        )

    def on_building_trainer_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.trainer_parameter_client = (
            builder._parameter_server_client,
            self.config.parameter_client_param_1,
        )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "parameter_client"


@dataclass
class MockExecutorDefaultConfig:
    executor_param: int = 1


class MockExecutor(Callback):
    def __init__(
        self,
        config: MockExecutorDefaultConfig = MockExecutorDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor(self, builder: SystemBuilder) -> None:
        """_summary_"""
        if builder._executor_id != "evaluator":
            builder.attr.exec = (
                builder._executor_id,
                builder.attr.adder,
                builder.attr.executor_parameter_client,
                self.config.executor_param,
            )
        else:
            builder.attr.exec = (
                builder._executor_id,
                builder.attr.executor_parameter_client,
                self.config.executor_param,
            )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "executor"


@dataclass
class MockExecutorEnvironmentLoopConfig:
    executor_environment_loop_param: str = "param"


class MockExecutorEnvironmentLoop(Callback):
    def __init__(
        self,
        config: MockExecutorEnvironmentLoopConfig = MockExecutorEnvironmentLoopConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.environment = (
            builder.attr.executor_logger,
            self.config.executor_environment_loop_param,
        )

    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """_summary_"""
        if builder._executor_id != "evaluator":
            builder.attr.system_executor = (
                builder._data_server_client,
                builder.attr.environment,
                builder.attr.exec,
            )
        else:
            builder.attr.system_executor = (
                builder.attr.environment,
                builder.attr.exec,
            )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "executor_environment_loop"


@dataclass
class MockTrainerDatasetConfig:
    trainer_dataset_param: int = 5


class MockTrainerDataset(Callback):
    def __init__(
        self,
        config: MockTrainerDatasetConfig = MockTrainerDatasetConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.dataset = self.config.trainer_dataset_param

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "trainer_dataset"


@dataclass
class MockTrainerConfig:
    trainer_param_0: int = 2
    trainer_param_1: str = "train"


class MockTrainer(Callback):
    def __init__(
        self,
        config: MockTrainerConfig = MockTrainerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_trainer(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.system_trainer = (
            builder._trainer_id,
            builder.attr.trainer_logger,
            builder.attr.dataset,
            builder.attr.trainer_parameter_client,
        )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "trainer"


@dataclass
class DistributorConfig:
    num_executors: int = 1
    nodes_on_gpu: List[str] = field(default_factory=list)
    multi_process: bool = True
    name: str = "system"


class MockDistributor(Callback):
    def __init__(self, config: DistributorConfig = DistributorConfig()) -> None:
        """Mock system distributor component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """_summary_"""
        data_server = builder.data_server()
        parameter_server = builder.parameter_server()

        trainer = builder.trainer(
            trainer_id="trainer",
            data_server_client=data_server,
            parameter_server_client=parameter_server,
        )
        executor = builder.executor(
            executor_id="executor",
            data_server_client=data_server,
            parameter_server_client=parameter_server,
        )
        evaluator = builder.executor(
            executor_id="evaluator",
            data_server_client=None,
            parameter_server_client=parameter_server,
        )
        builder.attr.system_build = (
            data_server,
            parameter_server,
            executor,
            evaluator,
            trainer,
        )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'.

        Returns:
            Component type name
        """
        return "distributor"
