# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

"""Unit tests for parameter client components"""

from types import SimpleNamespace

import pytest

from mava.components.jax.building.parameter_client import (
    ExecutorParameterClient,
    ExecutorParameterClientConfig,
    TrainerParameterClient,
)
from mava.systems.jax.builder import Builder
from mava.systems.jax.parameter_server import ParameterServer


@pytest.fixture
def mock_builder_with_parameter_client() -> Builder:
    """Create a mock builder for testing that has a parameter client."""

    builder = Builder(components=[])

    builder.store.networks = {
        "networks": {
            "network_agent_0": SimpleNamespace(params={"weights": 0, "biases": 0}),
            "network_agent_1": SimpleNamespace(params={"weights": 1, "biases": 1}),
            "network_agent_2": SimpleNamespace(params={"weights": 2, "biases": 2}),
        }
    }

    builder.store.trainer_networks = {
        "trainer_0": ["network_agent_0", "network_agent_1", "network_agent_2"]
    }
    builder.store.trainer_id = "trainer_0"

    builder.store.parameter_server_client = ParameterServer(
        store=SimpleNamespace(get_parameters={"key_1": 1}),
        components=[],
    )

    return builder


def test_executor_parameter_client_no_evaluator_with_parameter_client(
    mock_builder_with_parameter_client: Builder,
) -> None:
    """Test executor parameter client.

    Args:
        mock_builder_with_parameter_client : mava builder object
    """

    mock_builder = mock_builder_with_parameter_client
    mock_builder.store.is_evaluator = False
    exec_param_client = ExecutorParameterClient(
        config=ExecutorParameterClientConfig(executor_parameter_update_period=100)
    )
    exec_param_client.on_building_executor_parameter_client(builder=mock_builder)

    assert hasattr(mock_builder.store, "executor_parameter_client")
    assert mock_builder.store.executor_parameter_client._get_keys == [
        "networks-network_agent_0",
        "networks-network_agent_1",
        "networks-network_agent_2",
        "trainer_steps",
        "trainer_walltime",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    ]
    assert mock_builder.store.executor_parameter_client._set_keys == [
        "executor_episodes",
        "executor_steps",
    ]
    assert list(mock_builder.store.executor_parameter_client._parameters.keys()) == [
        "networks-network_agent_0",
        "networks-network_agent_1",
        "networks-network_agent_2",
        "trainer_steps",
        "trainer_walltime",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    ]
    for i in range(2):
        assert mock_builder.store.executor_parameter_client._parameters[
            f"networks-network_agent_{i}"
        ] == {"weights": i, "biases": i}
    assert mock_builder.store.executor_parameter_client._get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._update_period == 100


def test_executor_parameter_client_evaluator_with_parameter_client(
    mock_builder_with_parameter_client: Builder,
) -> None:
    """Test evaluator parameter client.

    Args:
        mock_builder_with_parameter_client: mava builder object
    """

    mock_builder = mock_builder_with_parameter_client
    mock_builder.store.is_evaluator = True
    exec_param_client = ExecutorParameterClient(
        config=ExecutorParameterClientConfig(executor_parameter_update_period=100)
    )
    exec_param_client.on_building_executor_parameter_client(builder=mock_builder)

    assert hasattr(mock_builder.store, "executor_parameter_client")
    assert mock_builder.store.executor_parameter_client._get_keys == [
        "networks-network_agent_0",
        "networks-network_agent_1",
        "networks-network_agent_2",
        "trainer_steps",
        "trainer_walltime",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    ]
    assert mock_builder.store.executor_parameter_client._set_keys == [
        "evaluator_steps",
        "evaluator_episodes",
    ]
    assert list(mock_builder.store.executor_parameter_client._parameters.keys()) == [
        "networks-network_agent_0",
        "networks-network_agent_1",
        "networks-network_agent_2",
        "trainer_steps",
        "trainer_walltime",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    ]
    for i in range(2):
        assert mock_builder.store.executor_parameter_client._parameters[
            f"networks-network_agent_{i}"
        ] == {"weights": i, "biases": i}
    assert mock_builder.store.executor_parameter_client._get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._update_period == 100


def test_executor_parameter_client_with_no_parameter_client(
    mock_builder_with_parameter_client: Builder,
) -> None:
    """Test executor parameter server client when no parameter \
        server client is added to the builder"""

    mock_builder = mock_builder_with_parameter_client
    mock_builder.store.is_evaluator = True
    mock_builder.store.parameter_server_client = False
    exec_param_client = ExecutorParameterClient(
        config=ExecutorParameterClientConfig(executor_parameter_update_period=100)
    )

    exec_param_client.on_building_executor_parameter_client(mock_builder)

    assert mock_builder.store.executor_parameter_client is None


def test_trainer_parameter_client(
    mock_builder_with_parameter_client: Builder,
) -> None:
    """Test trainer parameter client.

    Args:
        mock_builder_with_parameter_client: mava builder object
    """

    mock_builder = mock_builder_with_parameter_client
    trainer_param_client = TrainerParameterClient()
    trainer_param_client.on_building_trainer_parameter_client(mock_builder)

    assert hasattr(mock_builder.store, "trainer_parameter_client")
    assert mock_builder.store.trainer_parameter_client._get_keys == [
        "trainer_steps",
        "trainer_walltime",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    ]
    assert mock_builder.store.trainer_parameter_client._set_keys == [
        "networks-network_agent_0",
        "networks-network_agent_1",
        "networks-network_agent_2",
    ]
    assert list(mock_builder.store.trainer_parameter_client._parameters.keys()) == [
        "networks-network_agent_0",
        "networks-network_agent_1",
        "networks-network_agent_2",
        "trainer_steps",
        "trainer_walltime",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    ]
    assert mock_builder.store.trainer_parameter_client._get_call_counter == 0
    assert mock_builder.store.trainer_parameter_client._set_call_counter == 0
    assert mock_builder.store.trainer_parameter_client._set_get_call_counter == 0
    assert mock_builder.store.trainer_parameter_client._update_period == 1


def test_trainer_parameter_client_with_no_parameter_client(
    mock_builder_with_parameter_client: Builder,
) -> None:
    """Test trainer parameter server client when no parameter \
        server client is added to the builder"""

    mock_builder = mock_builder_with_parameter_client
    mock_builder.store.parameter_server_client = False
    trainer_param_client = TrainerParameterClient()

    trainer_param_client.on_building_trainer_parameter_client(mock_builder)

    assert mock_builder.store.trainer_parameter_client is None
