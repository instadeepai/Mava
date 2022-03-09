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

# TODO(Arnu): remove at a later stage

"""Tests for config class for Jax-based Mava systems"""

from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import List

import pytest

from mava.callbacks import Callback
from mava.core_jax import SystemBuilder
from mava.systems.jax.system import System


# Mock components to feed to the builder
@dataclass
class MockDataServerAdderDefaultConfig:
    adder_param_0: int = 1
    adder_param_1: str = "1"


class MockDataServerAdder(Callback):
    def __init__(
        self,
        config: MockDataServerAdderDefaultConfig = MockDataServerAdderDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.adder_signature = self.config.adder_param_0

    def on_building_data_server_rate_limiter(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.rate_limiter = self.config.adder_param_1

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "data_server_adder"


@dataclass
class MockDataServerDefaultConfig:
    data_server_param_0: int = 1
    data_server_param_1: str = "1"


class MockDataServer(Callback):
    def __init__(
        self, config: MockDataServerDefaultConfig = MockDataServerDefaultConfig()
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_data_server(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.system_data_server = (
            builder.blocks.adder_signature,
            builder.blocks.rate_limiter,
            self.config.data_server_param_0,
            self.config.data_server_param_1,
        )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "data_server"


@dataclass
class MockParameterServerDefaultConfig:
    parameter_server_param_0: int = 1
    parameter_server_param_1: str = "1"


class MockParameterServer(Callback):
    def __init__(
        self,
        config: MockParameterServerDefaultConfig = MockParameterServerDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_parameter_server(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.system_parameter_server = (
            self.config.parameter_server_param_0,
            self.config.parameter_server_param_1,
        )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "parameter_server"


@dataclass
class MockExecutorAdderDefaultConfig:
    executor_adder_param_0: int = 1
    executor_adder_param_1: str = "1"


class MockExecutorAdder(Callback):
    def __init__(
        self,
        config: MockExecutorAdderDefaultConfig = MockExecutorAdderDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_adder_priority(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.adder_priority = self.config.executor_adder_param_0

    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.adder = (
            builder.blocks.adder_priority,
            self.config.executor_adder_param_1,
        )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "executor_adder"


@dataclass
class MockExecutorDefaultConfig:
    executor_param_0: int = 1
    executor_param_1: str = "1"
    executor_param_2: bool = True


class MockExecutor(Callback):
    def __init__(
        self,
        config: MockExecutorDefaultConfig = MockExecutorDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.exec_logger = self.config.executor_param_0

    def on_building_executor_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.exec_param_client = self.config.executor_param_1

    def on_building_executor(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.exec = (
            builder.blocks.adder,
            builder.blocks.exec_param_client,
        )

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.env = (
            builder.blocks.exec_logger,
            self.config.executor_param_2,
        )

    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.system_executor = (
            builder.blocks.env,
            builder.blocks.exec,
        )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "executor"


@dataclass
class MockTrainerDatasetDefaultConfig:
    trainer_dataset_param_0: int = 1


class MockTrainerDataset(Callback):
    def __init__(
        self,
        config: MockTrainerDatasetDefaultConfig = MockTrainerDatasetDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.dataset = self.config.trainer_dataset_param_0

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "trainer_dataset"


@dataclass
class MockTrainerDefaultConfig:
    trainer_param_0: int = 1
    trainer_param_1: str = "1"


class MockTrainer(Callback):
    def __init__(
        self,
        config: MockTrainerDefaultConfig = MockTrainerDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.train_logger = self.config.trainer_param_0

    def on_building_trainer_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.train_param_client = self.config.trainer_param_1

    def on_building_trainer(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.system_trainer = (
            builder.blocks.train_logger,
            builder.blocks.train_param_client,
        )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "trainer"


@dataclass
class DistributorDefaultConfig:
    num_executors: int = 1
    nodes_on_gpu: List[str] = field(default_factory=list)
    multi_process: bool = True
    name: str = "system"


class MockDistributor(Callback):
    def __init__(
        self, config: DistributorDefaultConfig = DistributorDefaultConfig()
    ) -> None:
        """Mock system distributor component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'.

        Returns:
            Component type name
        """
        return "distributor"


@dataclass
class MockProgramDefaultConfig:
    program_param_0: int = 1
    program_param_1: str = "1"


class MockProgramConstructor(Callback):
    def __init__(
        self,
        config: MockProgramDefaultConfig = MockProgramDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """_summary_"""
        data_server = builder.data_server()
        parameter_server = builder.parameter_server()
        executor = builder.executor(
            executor_id="executor",
            data_server_client=data_server,
            parameter_server_client=parameter_server,
        )
        trainer = builder.trainer(
            trainer_id="trainer",
            data_server_client=data_server,
            parameter_server_client=parameter_server,
        )
        builder.blocks.system_build = (
            data_server,
            parameter_server,
            executor,
            trainer,
            self.config.program_param_0,
            self.config.program_param_1,
        )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "program"


@dataclass
class MockLauncherDefaultConfig:
    launcher_param_0: int = 1
    launcher_param_1: str = "1"


class MockLauncher(Callback):
    def __init__(
        self,
        config: MockLauncherDefaultConfig = MockLauncherDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_launch(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.system_launcher = (
            builder.blocks.system_build,
            self.config.launcher_param_0,
            self.config.launcher_param_1,
        )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "launcher"


class TestSystem(System):
    def design(self) -> SimpleNamespace:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        components = SimpleNamespace(
            data_server_adder=MockDataServerAdder,
            data_server=MockDataServer,
            parameter_server=MockParameterServer,
            executor_adder=MockExecutorAdder,
            executor=MockExecutor,
            trainer_dataset=MockTrainerDataset,
            trainer=MockTrainer,
            distributor=MockDistributor,
            program=MockProgramConstructor,
            launcher=MockLauncher,
        )
        return components


@pytest.fixture
def test_system() -> System:
    """Dummy system with zero components."""
    return TestSystem()


def test_builder(
    test_system: System,
) -> None:
    """Test if system can launch without having had changed (configured) the default \
        config."""
    test_system.launch(num_executors=1, nodes_on_gpu=["process"])
    (
        data_server,
        parameter_server,
        executor,
        trainer,
        build_param_0,
        build_param_1,
    ) = test_system._builder.blocks.system_build
    assert build_param_0 == 1
    assert build_param_1 == "1"

    (
        adder_signature,
        rate_limiter,
        data_server_param_0,
        data_server_param_1,
    ) = data_server
    assert adder_signature == 1
    assert rate_limiter == "1"
    assert data_server_param_0 == 1
    assert data_server_param_1 == "1"

    param_server_param_0, param_server_param_1 = parameter_server
    assert param_server_param_0 == 1
    assert param_server_param_1 == "1"

    env, exec = executor
    exec_logger, exec_logger_param = env
    assert exec_logger_param is True
    assert exec_logger == 1
    exec_adder, exec_param_client = exec
    assert exec_param_client == "1"
    adder_priority, adder_param = exec_adder
    assert adder_param == "1"
    assert adder_priority == 1

    train_logger, train_param_client = trainer
    assert train_logger == 1
    assert train_param_client == "1"
