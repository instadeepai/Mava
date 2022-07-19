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

from typing import Any, Callable, List

import launchpad as lp
import pytest
from reverb import Client, item_selectors, pybind, rate_limiters
from reverb import server as reverb_server

from mava.systems.jax.launcher import Launcher, NodeType


@pytest.fixture
def mock_data_server_fn() -> Callable:
    """call data_server function"""

    def data_server() -> List[Any]:
        """data_server

        Returns:
                tables: fake table composed of reverb_server tables
        """
        return [
            reverb_server.Table(
                name="table_0",
                sampler=item_selectors.Prioritized(priority_exponent=1),
                remover=item_selectors.Fifo(),
                max_size=1000,
                rate_limiter=rate_limiters.MinSize(1),
            )
        ]

    return data_server


@pytest.fixture
def mock_parameter_server_fn() -> Callable:
    """call parameter_server function"""

    def parameter_server() -> str:
        """Fake parameter server function"""
        return "test_parameter_server"

    return parameter_server


@pytest.fixture
def mock_parameter_server_second_fn() -> Callable:
    """call the second parameter_server function"""

    def parameter_server_second() -> str:
        """Another fake parameter server function"""
        return "test_parameter_server_second_mock"

    return parameter_server_second


def test_initiator_multi_process() -> None:
    """Test the constructor of Launcher in the case of multi process"""
    launcher = Launcher(multi_process=True)

    assert launcher._multi_process == True
    assert launcher._name == "System"
    assert launcher._sp_trainer_period == 10
    assert launcher._sp_evaluator_period == 10
    assert launcher._terminal == "current_terminal"

    assert launcher._nodes_on_gpu == []
    assert isinstance(launcher._program, lp.Program)
    assert launcher._program._name == "System"
    assert launcher._program._groups == {}
    assert launcher._program._current_group == None

    assert not hasattr(launcher, "_nodes")
    assert not hasattr(launcher, "_node_dict")


def test_initiator_non_multi_process() -> None:
    """Test the constructor of Launcher in the case of one process"""
    launcher = Launcher(multi_process=False)

    assert launcher._multi_process == False
    assert launcher._name == "System"
    assert launcher._sp_trainer_period == 10
    assert launcher._sp_evaluator_period == 10
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


def test_add_multi_process(mock_data_server_fn: Callable) -> None:
    """Test add method in the Launcher for the case of multi process

    Args:
        mock_data_server_fn
    """
    launcher = Launcher(multi_process=True)
    data_server = launcher.add(
        mock_data_server_fn,
        node_type=NodeType.reverb,
        name="data_server_test",
    )

    assert list(launcher._program._groups.keys()) == ["data_server_test"]
    assert (
        launcher._program._groups["data_server_test"][-1]._priority_tables_fn
        == mock_data_server_fn
    )
    assert [data_server] == launcher._program._groups["data_server_test"][
        -1
    ]._created_handles

    assert not hasattr(launcher, "_nodes")
    assert not hasattr(launcher, "_node_dict")


def test_add_multi_process_two_add_calls(
    mock_data_server_fn: Callable, mock_parameter_server_fn: Callable
) -> None:
    """Test calling add more than one time method in the Launcher for the case of multi process

    Args:
        mock_data_server_fn
        mock_parameter_server_fn
    """
    launcher = Launcher(multi_process=True)
    data_server = launcher.add(
        mock_data_server_fn,
        node_type=NodeType.reverb,
        name="data_server_test",
    )
    parameter_server = launcher.add(
        mock_parameter_server_fn,
        node_type=NodeType.courier,
        name="parameter_server_test",
    )

    assert list(launcher._program._groups.keys()) == [
        "data_server_test",
        "parameter_server_test",
    ]

    assert (
        launcher._program._groups["data_server_test"][-1]._priority_tables_fn
        == mock_data_server_fn
    )
    assert [data_server] == launcher._program._groups["data_server_test"][
        -1
    ]._created_handles

    assert (
        launcher._program._groups["parameter_server_test"][-1]._constructor
        == mock_parameter_server_fn
    )
    assert [parameter_server] == launcher._program._groups["parameter_server_test"][
        -1
    ]._created_handles

    assert not hasattr(launcher, "_nodes")
    assert not hasattr(launcher, "_node_dict")


def test_add_multi_process_two_add_same_name(
    mock_parameter_server_fn: Callable, mock_parameter_server_second_fn: Callable
) -> None:
    """Test calling twice add for two nodes with same name for the case of multi process

    Args:
        mock_parameter_server_fn
        mock_parameter_server_second_fn
    """
    launcher = Launcher(multi_process=True)

    parameter_server_1 = launcher.add(
        mock_parameter_server_fn,
        node_type=NodeType.courier,
        name="parameter_server_test",
    )
    parameter_server_2 = launcher.add(
        mock_parameter_server_second_fn,
        node_type=NodeType.courier,
        name="parameter_server_test",
    )

    assert list(launcher._program._groups.keys()) == ["parameter_server_test"]
    assert (
        launcher._program._groups["parameter_server_test"][0]._constructor
        == mock_parameter_server_fn
    )
    assert (
        launcher._program._groups["parameter_server_test"][1]._constructor
        == mock_parameter_server_second_fn
    )
    assert [parameter_server_1] == launcher._program._groups["parameter_server_test"][
        0
    ]._created_handles
    assert [parameter_server_2] == launcher._program._groups["parameter_server_test"][
        1
    ]._created_handles

    assert not hasattr(launcher, "_nodes")
    assert not hasattr(launcher, "_node_dict")


def test_add_non_multi_process_reverb_node(mock_data_server_fn: Callable) -> None:
    """Test the add method for the case of one process and for node_type reverb

    Args:
        mock_data_server_fn
    """
    launcher = Launcher(multi_process=False)
    with pytest.raises(ValueError):
        data_server = launcher.add(
            mock_data_server_fn,
            node_type=NodeType.reverb,
            name="data_server_test",
        )
    data_server = launcher.add(
        mock_data_server_fn,
        node_type=NodeType.reverb,
        name="data_server",
    )

    assert not hasattr(launcher, "_program")

    assert launcher._replay_server._port != None
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


def test_add_non_multi_process_courier_node(mock_parameter_server_fn: Callable) -> None:
    """Test the add method for the case of one process and for node_type courier

    Args:
        mock_parameter_server_fn
    """
    launcher = Launcher(multi_process=False)
    with pytest.raises(ValueError):
        parameter_server = launcher.add(
            mock_parameter_server_fn,
            node_type=NodeType.courier,
            name="parameter_server_test",
        )
    parameter_server = launcher.add(
        mock_parameter_server_fn,
        node_type=NodeType.courier,
        name="parameter_server",
    )

    assert not hasattr(launcher, "_program")
    assert not hasattr(launcher, "_replay_server")

    assert launcher._nodes[-1] == launcher._node_dict["parameter_server"]
    assert parameter_server == launcher._nodes[-1]

    assert parameter_server == "test_parameter_server"


def test_add_non_multi_process_two_add_calls(
    mock_data_server_fn: Callable, mock_parameter_server_fn: Callable
) -> None:
    """Test calling twice add method for the case of one process

    Args:
        mock_data_server_fn
        mock_parameter_server_fn
    """
    launcher = Launcher(multi_process=False)
    data_server = launcher.add(
        mock_data_server_fn,
        node_type=NodeType.reverb,
        name="data_server",
    )
    parameter_server = launcher.add(
        mock_parameter_server_fn,
        node_type=NodeType.courier,
        name="parameter_server",
    )

    assert not hasattr(launcher, "_program")

    assert launcher._nodes == [data_server, parameter_server]
    assert launcher._node_dict["data_server"] == data_server
    assert launcher._node_dict["parameter_server"] == parameter_server

    assert launcher._replay_server._port != None
    assert type(launcher._replay_server._port) == int
    assert isinstance(launcher._replay_server._server, pybind.Server)
    assert isinstance(launcher._node_dict["data_server"], Client)
    assert launcher._node_dict["data_server"]._server_address == (
        f"localhost:{launcher._replay_server._port}"
    )
    assert isinstance(launcher._node_dict["data_server"]._client, pybind.Client)
    assert launcher._node_dict["data_server"]._signature_cache == {}

    assert launcher._node_dict["parameter_server"] == "test_parameter_server"


def test_add_non_multi_process_two_add_same_name(
    mock_parameter_server_fn: Callable, mock_parameter_server_second_fn: Callable
) -> None:
    """Test calling twice add method for two nodes with same name and for the case of one process and for node_type reverb

    Args:
        mock_data_server_fn
    """
    launcher = Launcher(multi_process=False)

    parameter_server_1 = launcher.add(
        mock_parameter_server_fn,
        node_type=NodeType.courier,
        name="parameter_server",
    )
    with pytest.raises(ValueError):
        parameter_server_2 = launcher.add(
            mock_parameter_server_second_fn,
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


def test_get_nodes_non_multi_process(
    mock_data_server_fn: Callable, mock_parameter_server_fn: Callable
) -> None:
    """Test get_nodes method in the case of one process with two nodes

    Args:
        mock_data_server_fn
        mock_parameter_server_fn
    """
    launcher = Launcher(multi_process=False)
    data_server = launcher.add(
        mock_data_server_fn,
        node_type=NodeType.reverb,
        name="data_server",
    )
    parameter_server = launcher.add(
        mock_parameter_server_fn,
        node_type=NodeType.courier,
        name="parameter_server",
    )

    nodes = launcher.get_nodes()

    assert nodes == [data_server, parameter_server]
