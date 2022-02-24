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


"""Tests for core Mava interfaces for Jax systems."""

from types import SimpleNamespace
from typing import Any, Dict, List

import pytest

from mava.core_jax import BaseSystem, SystemBuilder

# Dummy component
COMPONENT = None


class TestDummySystem(BaseSystem):
    """Create a complete class with all abstract method overwritten."""

    def design(self) -> SimpleNamespace:
        """Dummy design"""
        self.components = SimpleNamespace(component_0=COMPONENT, component_1=COMPONENT)
        assert self.components.component_0 is None
        assert self.components.component_1 is None
        return self.components

    def update(self, component: Any) -> None:
        """Dummy update"""
        assert component is None

    def add(self, component: Any) -> None:
        """Dummy add"""
        assert component is None

    def configure(self, **kwargs: Any) -> None:
        """Dummy configure"""
        assert kwargs["param_0"] == 0
        assert kwargs["param_1"] == 1

    def launch(
        self,
        num_executors: int,
        nodes_on_gpu: List[str],
        multi_process: bool = True,
        name: str = "system",
    ) -> None:
        """Dummy launch"""
        assert num_executors == 1
        assert multi_process is True
        assert nodes_on_gpu[0] == "process"
        assert name == "system"


@pytest.fixture
def dummy_system() -> TestDummySystem:
    """Create complete system for use in tests"""
    return TestDummySystem()


def test_dummy_system(dummy_system: TestDummySystem) -> None:
    """Test complete system methods"""
    # design system
    dummy_system.design()

    # update component
    dummy_system.update(COMPONENT)

    # add component
    dummy_system.add(COMPONENT)

    # configure system
    dummy_system.configure(param_0=0, param_1=1)

    # launch system
    dummy_system.launch(num_executors=1, nodes_on_gpu=["process"])


# Dummy variables
DATA_CLIENT = "data_client"
PARAMETER_CLIENT = "parameter_client"


class TestDummySystemBuilder(SystemBuilder):
    """Create a complete class with all abstract method overwritten."""

    def data_server(self) -> List[Any]:
        """Dummy Data server."""
        data_server = ["data_server"]
        return data_server

    def parameter_server(self, extra_nodes: Dict = {}) -> Any:
        """Dummy Parameter server."""
        parameter_server = ("parameter_server", extra_nodes)
        return parameter_server

    def executor(
        self, executor_id: Any, data_server_client: Any, parameter_server_client: Any
    ) -> Any:
        """Dummy Executor."""
        executor = (executor_id, data_server_client, parameter_server_client)
        return executor

    def trainer(
        self, trainer_id: Any, data_server_client: Any, parameter_server_client: Any
    ) -> Any:
        """Dummy Trainer"""
        trainer = (trainer_id, data_server_client, parameter_server_client)
        return trainer


@pytest.fixture
def dummy_system_builder() -> TestDummySystemBuilder:
    """Create complete system for use in tests"""
    return TestDummySystemBuilder()


def test_dummy_system_builder(dummy_system_builder: TestDummySystemBuilder) -> None:
    """Test complete system methods"""
    # build data server
    data_server = dummy_system_builder.data_server()
    assert data_server[0] == "data_server"

    # build parameter server
    parameter_server, extra_nodes = dummy_system_builder.parameter_server()
    assert parameter_server == "parameter_server"
    assert extra_nodes == {}

    # build executor
    exec_id, exec_data_client, exec_param_client = dummy_system_builder.executor(
        executor_id="executor",
        data_server_client=DATA_CLIENT,
        parameter_server_client=PARAMETER_CLIENT,
    )

    assert exec_id == "executor"
    assert exec_data_client == "data_client"
    assert exec_param_client == "parameter_client"

    # build trainer
    train_id, train_data_client, train_param_client = dummy_system_builder.trainer(
        trainer_id="trainer",
        data_server_client=DATA_CLIENT,
        parameter_server_client=PARAMETER_CLIENT,
    )

    assert train_id == "trainer"
    assert train_data_client == "data_client"
    assert train_param_client == "parameter_client"
