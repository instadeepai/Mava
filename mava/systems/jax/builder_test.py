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
    adder_param_1: str = "default"


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
    data_server_param_0: float = 2.7
    data_server_param_1: bool = False


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
    parameter_server_param_0: int = 2
    parameter_server_param_1: str = "setting"


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
    executor_adder_param_0: float = 23.2
    executor_adder_param_1: str = "random"


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
    executor_param_1: str = "param"
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
        builder.blocks.exec_param_client = builder._parameter_server_client

    # TODO(Arnu): handle this using a decorator
    def on_building_executor(self, builder: SystemBuilder) -> None:
        """_summary_"""
        if builder._executor_id != "evaluator":
            builder.blocks.exec = (
                builder._executor_id,
                builder.blocks.adder,
                builder.blocks.exec_param_client,
            )
        else:
            builder.blocks.exec = (
                builder._executor_id,
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
        if builder._executor_id != "evaluator":
            builder.blocks.system_executor = (
                builder._data_server_client,
                builder.blocks.env,
                builder.blocks.exec,
            )
        else:
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
    trainer_dataset_param_0: int = 5


class MockTrainerDataset(Callback):
    def __init__(
        self,
        config: MockTrainerDatasetDefaultConfig = MockTrainerDatasetDefaultConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.dataset = (
            builder._table_name,
            self.config.trainer_dataset_param_0,
        )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "trainer_dataset"


@dataclass
class MockTrainerDefaultConfig:
    trainer_param_0: int = 2
    trainer_param_1: str = "train"


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
        builder.blocks.train_param_client = builder._parameter_server_client

    def on_building_trainer(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.blocks.system_trainer = (
            builder._trainer_id,
            builder.blocks.train_logger,
            builder.blocks.dataset,
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
    program_param: str = "name"


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
        evaluator = builder.executor(
            executor_id="evaluator",
            data_server_client=None,
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
            evaluator,
            trainer,
            self.config.program_param,
        )

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "program"


@dataclass
class MockLauncherDefaultConfig:
    launcher_param: str = "name"


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
            self.config.launcher_param,
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
    """Test if system builder instantiates processes as expected."""
    test_system.launch(num_executors=1, nodes_on_gpu=["process"])
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
        build_param,
    ) = test_system._builder.blocks.system_build
    assert build_param == "name"
    assert data_server == (1, "default", 2.7, False)
    assert parameter_server == (2, "setting")

    exec_data_client, env, exec = executor
    exec_logger, exec_logger_param = env
    assert exec_data_client == data_server
    assert exec_logger_param is True
    assert exec_logger == 1

    exec_id, exec_adder, exec_param_client = exec
    assert exec_id == "executor"
    assert exec_param_client == parameter_server
    assert exec_adder == (23.2, "random")

    eval_env, eval_exec = evaluator
    eval_id, eval_param_client = eval_exec
    assert eval_env == env
    assert eval_id == "evaluator"
    assert eval_param_client == exec_param_client

    train_id, train_logger, train_dataset, train_param_client = trainer
    assert train_id == "trainer"
    assert train_logger == 2
    assert train_dataset == ("table_trainer", 5)
    assert train_param_client == parameter_server
