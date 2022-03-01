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

# TODO (Arnu): remove when attribute errors for builder and mixin have been figured out.
# type: ignore

"""Tests for Jax-based Mava system implementation."""
from dataclasses import dataclass
from typing import Any, List

import pytest

from mava.callbacks import Callback, CallbackHookMixin
from mava.core_jax import SystemBuilder


# TODO (Arnu): create reusable mocked classes and fixtures using conftest
# Mock Mixin
class MockDataServerBuilder(SystemBuilder, CallbackHookMixin):
    def __init__(
        self,
        components: List[Any],
    ) -> None:
        """System building init

        Args:
            components: system callback components
        """

        self.callbacks = components

    def data_server(self) -> List[Any]:
        """Data server to store and serve transition data from and to system.

        Returns:
            System data server
        """

        # start of make replay tables
        self.on_building_data_server_start()

        # make tables
        self.on_building_data_server()

        # end of make replay tables
        self.on_building_data_server_end()

        return self.system_data_server

    def parameter_server(self) -> Any:
        """Parameter server to store and serve system network parameters."""
        pass

    def executor(
        self, executor_id: str, data_server_client: Any, parameter_server_client: Any
    ) -> Any:
        """Executor, a collection of agents in an environment to gather experience."""
        pass

    def trainer(
        self, trainer_id: str, data_server_client: Any, parameter_server_client: Any
    ) -> Any:
        """Trainer, a system process for updating agent specific network parameters."""
        pass

    def build(self) -> None:
        """Construct program nodes."""
        pass

    def launch(self) -> None:
        """Run the graph program."""
        pass


# Mock components
@dataclass
class MockDefaultConfig:
    param_0: int = 1
    param_1: str = "1"


class MockDataServerComponent(Callback):
    def __init__(self, config: MockDefaultConfig = MockDefaultConfig()) -> None:
        """Mock system component.

        Args:
            config : dataclass configuration for setting component hyperparameters
        """
        self.config = config

    def on_building_data_server_start(self, builder: SystemBuilder) -> None:
        """Dummy component function.

        Returns:
            config int plus string cast to int
        """
        builder.data_server_start = "start"

    def on_building_data_server(self, builder: SystemBuilder) -> None:
        """Dummy component function.

        Returns:
            config int plus string cast to int
        """
        data_server = (
            builder.data_server_start,
            self.config.param_0,
            self.config.param_1,
        )
        builder.system_data_server = data_server

    def on_building_data_end(self, builder: SystemBuilder) -> None:
        """Dummy component function.

        Returns:
            config int plus string cast to int
        """
        builder.system_data_server = (*builder.system_data_server, "end")

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'.

        Returns:
            Component type name
        """
        return "data_server"


@pytest.fixture
def mock_data_server_builder():
    """Mock builder with data server component."""
    components = [MockDataServerComponent]
    return MockDataServerBuilder(components=components)


# Tests
def test_mock_data_server_creation(
    mock_data_server_builder: MockDataServerBuilder,
) -> None:
    """Test builder hook calls using mixin.

    Args:
        mock_data_server_builder : mock builder
    """
    mock_data_server_builder.data_server()
    start, param_0, param_1, end = mock_data_server_builder.system_data_server
    assert start == "start"
    assert param_0 == 1
    assert param_1 == "1"
    assert end == "end"
