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

"""Tests for Distributor class for Jax-based Mava systems"""

from types import SimpleNamespace
from typing import Any, List

import pytest
from reverb import item_selectors, rate_limiters
from reverb import server as reverb_server

from mava.components.jax.building.distributor import Distributor
from mava.systems.jax.builder import Builder
from mava.systems.jax.launcher import Launcher


class MockBuilder(Builder):
    def __init__(self) -> None:
        """Initiator of a MockBuilder

        Attributes:
            trainer_network
            program: used to test on_building_launch method
            test_launch: used to test on_building_launch method
            test: used to test on_building_program_nodes method
            test_values: used to test on_building_program_nodes method
        """
        trainer_networks = {"trainer": ["network_agent"]}
        program = SimpleNamespace(launch=self.launch)
        store = SimpleNamespace(trainer_networks=trainer_networks, program=program)
        self.store = store
        self.node_built = {
            "data_server": False,
            "parameter_server": False,
            "executor": False,
            "evaluator": False,
            "trainer": False,
        }
        self.built_node_values = {}  # type: ignore
        self.program_launched = False

    def data_server(self) -> List[Any]:
        """Data server to test no multi_process in on_building_program_nodes method

        Returns:
            tables: fake table composed of reverb_server tables
        """
        self.node_built["data_server"] = True
        return [
            reverb_server.Table(
                name="table_0",
                sampler=item_selectors.Prioritized(priority_exponent=1),
                remover=item_selectors.Fifo(),
                max_size=1000,
                rate_limiter=rate_limiters.MinSize(1),
            )
        ]

    def parameter_server(self) -> str:
        """parameter_server to test no multi_process in on_building_program_nodes"""
        self.node_built["parameter_server"] = True
        return "Parameter Server Test"

    def executor(
        self, executor_id: str, data_server_client: Any, parameter_server_client: Any
    ) -> None:
        """Executor to test no multi_process in on_building_program_nodes method"""
        if executor_id == "evaluator":
            self.node_built["evaluator"] = True
        else:
            self.node_built["executor"] = True
        self.built_node_values["data_server_client"] = data_server_client
        self.built_node_values["parameter_server_client"] = parameter_server_client

    def trainer(
        self, trainer_id: str, data_server_client: Any, parameter_server_client: Any
    ) -> None:
        """Trainer to test no multi_process in on_building_program_nodes method"""
        self.node_built["trainer"] = True
        self.built_node_values["data_server_client"] = data_server_client
        self.built_node_values["parameter_server_client"] = parameter_server_client

    def launch(self) -> None:
        """Launch to test on_building_launch method"""
        self.program_launched = True


@pytest.fixture
def mock_builder() -> MockBuilder:
    """Mock builder"""
    return MockBuilder()


@pytest.fixture
def distributor() -> Distributor:
    """Distributor fixture for testing"""
    return Distributor()


def test_initiator(distributor: Distributor) -> None:
    """Test initiator of the Distributor"""
    assert distributor.config.nodes_on_gpu == ["trainer"]


def test_on_building_program_nodes_multi_process(
    mock_builder: MockBuilder, distributor: Distributor
) -> None:
    """Test on_building_program_nodes for multi_process distributor"""
    distributor.on_building_program_nodes(builder=mock_builder)

    assert isinstance(mock_builder.store.program, Launcher)

    assert list(mock_builder.store.program._program._groups.keys()) == [
        "data_server",
        "parameter_server",
        "executor",
        "evaluator",
        "trainer",
    ]
    assert (
        mock_builder.store.program._program._groups["data_server"][
            -1
        ]._priority_tables_fn
        == mock_builder.data_server
    )
    assert (
        mock_builder.store.program._program._groups["parameter_server"][-1]._constructor
        == mock_builder.parameter_server
    )
    assert (
        mock_builder.store.program._program._groups["executor"][-1]._constructor
        == mock_builder.executor
    )
    assert (
        mock_builder.store.program._program._groups["evaluator"][-1]._constructor
        == mock_builder.executor
    )
    assert (
        mock_builder.store.program._program._groups["trainer"][-1]._constructor
        == mock_builder.trainer
    )

    with pytest.raises(Exception):
        mock_builder.store.program.get_nodes()


def test_on_building_program_nodes_multi_process_no_evaluator(
    mock_builder: MockBuilder,
    distributor: Distributor,
) -> None:
    """Test on_building_program_nodes, multi_process distributor, no evaluator runs"""
    distributor.config.run_evaluator = False
    distributor.on_building_program_nodes(builder=mock_builder)

    assert isinstance(mock_builder.store.program, Launcher)

    assert list(mock_builder.store.program._program._groups.keys()) == [
        "data_server",
        "parameter_server",
        "executor",
        "trainer",
    ]
    assert (
        mock_builder.store.program._program._groups["data_server"][
            -1
        ]._priority_tables_fn
        == mock_builder.data_server
    )
    assert (
        mock_builder.store.program._program._groups["parameter_server"][-1]._constructor
        == mock_builder.parameter_server
    )
    assert (
        mock_builder.store.program._program._groups["executor"][-1]._constructor
        == mock_builder.executor
    )
    assert (
        mock_builder.store.program._program._groups["trainer"][-1]._constructor
        == mock_builder.trainer
    )

    with pytest.raises(Exception):
        mock_builder.store.program.get_nodes()


def test_on_building_program_nodes(
    mock_builder: MockBuilder, distributor: Distributor
) -> None:
    """Test on_building_program_nodes for non multi_process distributor"""
    distributor.config.multi_process = False
    distributor.config.run_evaluator = True
    distributor.on_building_program_nodes(builder=mock_builder)

    assert isinstance(mock_builder.store.program, Launcher)

    for value in mock_builder.node_built.values():
        assert value
    assert (
        mock_builder.built_node_values["data_server_client"]
        == mock_builder.store.program._node_dict["data_server"]
    )
    assert (
        mock_builder.built_node_values["parameter_server_client"]
        == "Parameter Server Test"
    )

    assert mock_builder.store.system_build == mock_builder.store.program._nodes


def test_on_building_program_nodes_no_evaluator(
    mock_builder: MockBuilder, distributor: Distributor
) -> None:
    """Test on_building_program_nodes, single process distributor, no evaluator runs"""
    distributor.config.multi_process = False
    distributor.config.run_evaluator = False
    distributor.on_building_program_nodes(builder=mock_builder)

    assert isinstance(mock_builder.store.program, Launcher)

    for key, value in mock_builder.node_built.items():
        if key == "evaluator":
            assert not value
        else:
            assert value
    assert (
        mock_builder.built_node_values["data_server_client"]
        == mock_builder.store.program._node_dict["data_server"]
    )
    assert (
        mock_builder.built_node_values["parameter_server_client"]
        == "Parameter Server Test"
    )

    assert mock_builder.store.system_build == mock_builder.store.program._nodes


def test_on_building_launch(
    mock_builder: MockBuilder, distributor: Distributor
) -> None:
    """Test on_building_launch"""
    distributor.on_building_launch(builder=mock_builder)
    assert mock_builder.program_launched
