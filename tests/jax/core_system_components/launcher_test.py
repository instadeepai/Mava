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

"""Tests for launcher class for Jax-based Mava systems"""

import launchpad as lp
import pytest
from reverb import Client, pybind

from mava.systems.launcher import Launcher, NodeType
from tests.jax.components.building.distributor_test import MockBuilder


@pytest.fixture
def mock_builder() -> MockBuilder:
    """Mock builder"""
    return MockBuilder()


def test_initiator_multi_process() -> None:
    """Test the constructor of Launcher in the case of multi process"""
    launcher = Launcher(multi_process=True)

    assert launcher._multi_process is True
    assert launcher._name == "System"
    assert launcher._single_process_trainer_period == 1
    assert launcher._single_process_evaluator_period == 10
    assert launcher._terminal == "current_terminal"

    assert launcher._nodes_on_gpu == []
    assert isinstance(launcher._program, lp.Program)
    assert launcher._program._name == "System"
    assert launcher._program._groups == {}
    assert launcher._program._current_group is None

    assert not hasattr(launcher, "_nodes")
    assert not hasattr(launcher, "_node_dict")


def test_initiator_non_multi_process() -> None:
    """Test the constructor of Launcher in the case of one process"""
    launcher = Launcher(multi_process=False)

    assert launcher._multi_process is False
    assert launcher._name == "System"
    assert launcher._single_process_trainer_period == 1
    assert launcher._single_process_evaluator_period == 10
    assert launcher._terminal == "current_terminal"

    assert launcher._nodes == []
    assert launcher._node_dict == {
        "data_server": None,
        "parameter_server": None,
        "executor": None,
        "evaluator": None,
        "trainer": None,
    }

    assert not hasattr(launcher, "_nodes_on_gpu")
    assert not hasattr(launcher, "_program")


def test_add_multi_process(mock_builder: MockBuilder) -> None:
    """Test add method in the Launcher for the case of multi process

    Args:
        mock_builder: Mock of the builder
    """
    launcher = Launcher(multi_process=True)
    data_server = launcher.add(
        mock_builder.data_server,
        node_type=NodeType.reverb,
        name="data_server_test",
    )

    assert list(launcher._program._groups.keys()) == ["data_server_test"]

    # Make sure the node have data_server method
    data_server_fn = launcher._program._groups["data_server_test"][
        -1
    ]._priority_tables_fn
    assert str(repr(data_server_fn).split(" ")[2].split(".")[-1]) == "data_server"

    assert [data_server] == launcher._program._groups["data_server_test"][
        -1
    ]._created_handles

    assert not hasattr(launcher, "_nodes")
    assert not hasattr(launcher, "_node_dict")


def test_add_multi_process_two_add_calls(mock_builder: MockBuilder) -> None:
    """Test calling add more than one time method in the Launcher for the case of multi process # noqa:E501

    Args:
        mock_builder: Mock of the builder
    """
    launcher = Launcher(multi_process=True)
    data_server = launcher.add(
        mock_builder.data_server,
        node_type=NodeType.reverb,
        name="data_server_test",
    )
    parameter_server = launcher.add(
        mock_builder.parameter_server,
        node_type=NodeType.courier,
        name="parameter_server_test",
    )

    assert list(launcher._program._groups.keys()) == [
        "data_server_test",
        "parameter_server_test",
    ]

    # Make sure the node have data_server method
    data_server_fn = launcher._program._groups["data_server_test"][
        -1
    ]._priority_tables_fn
    assert str(repr(data_server_fn).split(" ")[2].split(".")[-1]) == "data_server"

    assert [data_server] == launcher._program._groups["data_server_test"][
        -1
    ]._created_handles

    assert (
        launcher._program._groups["parameter_server_test"][-1]._constructor()
        == "Parameter Server Test"
    )
    assert [parameter_server] == launcher._program._groups["parameter_server_test"][
        -1
    ]._created_handles

    assert not hasattr(launcher, "_nodes")
    assert not hasattr(launcher, "_node_dict")


def test_add_multi_process_two_add_same_name(mock_builder: MockBuilder) -> None:
    """Test calling twice add for two nodes with same name for the case of multi process

    Args:
        mock_builder: mock of the builder
    """
    launcher = Launcher(multi_process=True)

    parameter_server_1 = launcher.add(
        mock_builder.parameter_server,
        node_type=NodeType.courier,
        name="parameter_server_test",
    )
    parameter_server_2 = launcher.add(
        mock_builder.parameter_server,
        node_type=NodeType.courier,
        name="parameter_server_test",
    )

    assert list(launcher._program._groups.keys()) == ["parameter_server_test"]
    assert (
        launcher._program._groups["parameter_server_test"][0]._constructor()
        == "Parameter Server Test"
    )
    assert (
        launcher._program._groups["parameter_server_test"][1]._constructor()
        == "Parameter Server Test"
    )
    assert [parameter_server_1] == launcher._program._groups["parameter_server_test"][
        0
    ]._created_handles
    assert [parameter_server_2] == launcher._program._groups["parameter_server_test"][
        1
    ]._created_handles

    assert not hasattr(launcher, "_nodes")
    assert not hasattr(launcher, "_node_dict")


def test_add_non_multi_process_reverb_node(mock_builder: MockBuilder) -> None:
    """Test the add method for the case of one process and for node_type reverb

    Args:
        mock_builder: mock of the builder
    """
    launcher = Launcher(multi_process=False)
    with pytest.raises(ValueError):
        data_server = launcher.add(
            mock_builder.data_server,
            node_type=NodeType.reverb,
            name="data_server_test",
        )
    data_server = launcher.add(
        mock_builder.data_server,
        node_type=NodeType.reverb,
        name="data_server",
    )

    assert not hasattr(launcher, "_program")

    assert launcher._replay_server._port is not None
    assert type(launcher._replay_server._port) == int
    assert isinstance(launcher._replay_server._server, pybind.Server)

    assert launcher._nodes[-1] == launcher._node_dict["data_server"]
    assert data_server == launcher._nodes[-1]

    assert isinstance(launcher._node_dict["data_server"], Client)
    assert launcher._node_dict["data_server"]._server_address == (
        f"localhost:{launcher._replay_server._port}"
    )
    assert isinstance(launcher._node_dict["data_server"]._client, pybind.Client)
    assert launcher._node_dict["data_server"]._signature_cache == {}


def test_add_non_multi_process_courier_node(mock_builder: MockBuilder) -> None:
    """Test the add method for the case of one process and for node_type courier

    Args:
        mock_builder: mock of the builder
    """
    launcher = Launcher(multi_process=False)
    with pytest.raises(ValueError):
        parameter_server = launcher.add(
            mock_builder.parameter_server,
            node_type=NodeType.courier,
            name="parameter_server_test",
        )
    parameter_server = launcher.add(
        mock_builder.parameter_server,
        node_type=NodeType.courier,
        name="parameter_server",
    )

    assert not hasattr(launcher, "_program")
    assert not hasattr(launcher, "_replay_server")

    assert launcher._nodes[-1] == launcher._node_dict["parameter_server"]
    assert parameter_server == launcher._nodes[-1]

    assert parameter_server == "Parameter Server Test"


def test_add_non_multi_process_two_add_calls(mock_builder: MockBuilder) -> None:
    """Test calling twice add method for the case of one process

    Args:
        mock_builder: mock of the builder
    """
    launcher = Launcher(multi_process=False)
    data_server = launcher.add(
        mock_builder.data_server,
        node_type=NodeType.reverb,
        name="data_server",
    )
    parameter_server = launcher.add(
        mock_builder.parameter_server,
        node_type=NodeType.courier,
        name="parameter_server",
    )

    assert not hasattr(launcher, "_program")

    assert launcher._nodes == [data_server, parameter_server]
    assert launcher._node_dict["data_server"] == data_server
    assert launcher._node_dict["parameter_server"] == parameter_server

    assert launcher._replay_server._port is not None
    assert type(launcher._replay_server._port) == int
    assert isinstance(launcher._replay_server._server, pybind.Server)
    assert isinstance(launcher._node_dict["data_server"], Client)
    assert launcher._node_dict["data_server"]._server_address == (
        f"localhost:{launcher._replay_server._port}"
    )
    assert isinstance(launcher._node_dict["data_server"]._client, pybind.Client)
    assert launcher._node_dict["data_server"]._signature_cache == {}

    assert launcher._node_dict["parameter_server"] == "Parameter Server Test"


def test_add_non_multi_process_two_add_same_name(mock_builder: MockBuilder) -> None:
    """Test calling twice add method for two nodes with same name and for the case of one process and for node_type reverb # noqa:E501

    Args:
        mock_builder: mock of the builder
    """
    launcher = Launcher(multi_process=False)

    _ = launcher.add(
        mock_builder.parameter_server,
        node_type=NodeType.courier,
        name="parameter_server",
    )
    with pytest.raises(ValueError):
        _ = launcher.add(
            mock_builder.parameter_server,
            node_type=NodeType.courier,
            name="parameter_server",
        )


def test_get_nodes_multi_process() -> None:
    """Test get_nodes method in case of multi process"""
    launcher = Launcher(multi_process=True)
    with pytest.raises(ValueError):
        launcher.get_nodes()


def test_get_nodes_non_multi_process_empty() -> None:
    """Test get_nodes method in the case of one process with empty nodes list"""
    launcher = Launcher(multi_process=False)
    nodes = launcher.get_nodes()

    assert nodes == []


def test_get_nodes_non_multi_process(mock_builder: MockBuilder) -> None:
    """Test get_nodes method in the case of one process with two nodes

    Args:
        mock_builder: mock of the builder
    """
    launcher = Launcher(multi_process=False)
    data_server = launcher.add(
        mock_builder.data_server,
        node_type=NodeType.reverb,
        name="data_server",
    )
    parameter_server = launcher.add(
        mock_builder.parameter_server,
        node_type=NodeType.courier,
        name="parameter_server",
    )

    nodes = launcher.get_nodes()

    assert nodes == [data_server, parameter_server]
