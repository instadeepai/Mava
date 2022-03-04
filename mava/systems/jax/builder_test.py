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

from dataclasses import dataclass

from mava.callbacks import Callback
from mava.core_jax import SystemBuilder

# import pytest


# from mava.systems.jax import Builder, Config


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
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_data_server_adder_signature(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass

    def on_building_data_server_rate_limiter(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass


@dataclass
class MockDataServerDefaultConfig:
    data_server_param_0: int = 1
    data_server_param_1: str = "1"


class MockDataServer(Callback):
    def __init__(
        self, config: MockDataServerDefaultConfig = MockDataServerDefaultConfig()
    ) -> None:
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_data_server(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass


@dataclass
class MockParameterServerDefaultConfig:
    parameter_server_param_0: int = 1
    parameter_server_param_1: str = "1"


class MockParameterServerAdder(Callback):
    def __init__(
        self,
        config: MockParameterServerDefaultConfig = MockParameterServerDefaultConfig(),
    ) -> None:
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_parameter_server(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass


@dataclass
class MockExecutorAdderDefaultConfig:
    executor_adder_param_0: int = 1
    executor_adder_param_1: str = "1"


class MockExecutorAdder(Callback):
    def __init__(
        self,
        config: MockExecutorAdderDefaultConfig = MockExecutorAdderDefaultConfig(),
    ) -> None:
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_executor_adder_priority(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass

    def on_building_executor_adder(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass


@dataclass
class MockExecutorDefaultConfig:
    executor_param_0: int = 1
    executor_param_1: str = "1"


class MockExecutor(Callback):
    def __init__(
        self,
        config: MockExecutorDefaultConfig = MockExecutorDefaultConfig(),
    ) -> None:
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass

    def on_building_executor_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass

    def on_building_executor(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass

    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass


@dataclass
class MockTrainerDatasetDefaultConfig:
    trainer_dataset_param_0: int = 1
    trainer_dataset_param_1: str = "1"


class MockTrainerDataset(Callback):
    def __init__(
        self,
        config: MockTrainerDatasetDefaultConfig = MockTrainerDatasetDefaultConfig(),
    ) -> None:
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass


@dataclass
class MockTrainerDefaultConfig:
    trainer_param_0: int = 1
    trainer_param_1: str = "1"


class MockTrainer(Callback):
    def __init__(
        self,
        config: MockTrainerDefaultConfig = MockTrainerDefaultConfig(),
    ) -> None:
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass

    def on_building_trainer_parameter_client(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass

    def on_building_trainer(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass


@dataclass
class MockProgramDefaultConfig:
    program_param_0: int = 1
    program_param_1: str = "1"


class MockProgramConstructor(Callback):
    def __init__(
        self,
        config: MockProgramDefaultConfig = MockProgramDefaultConfig(),
    ) -> None:
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass


@dataclass
class MockLauncherDefaultConfig:
    launcher_param_0: int = 1
    launcher_param_1: str = "1"


class MockLauncher(Callback):
    def __init__(
        self,
        config: MockLauncherDefaultConfig = MockLauncherDefaultConfig(),
    ) -> None:
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_launch(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass
