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
from typing import Any

import numpy as np
import pytest

from mava.components.jax.building.parameter_client import (
    BaseParameterClient,
    ExecutorParameterClient,
    ExecutorParameterClientConfig,
    ExecutorParameterClientSeparateNetworks,
    TrainerParameterClient,
    TrainerParameterClientSeparateNetworks,
)
from mava.systems.jax.builder import Builder
from mava.systems.jax.parameter_server import ParameterServer

##############
# SHARED DATA
##############


class MockBaseParameterClient(BaseParameterClient):
    def __init__(self, config: Any) -> None:
        """Initialize mock base parameter client class to test the \
            _set_up_count_parameters method in BaseParameterClient."""
        super().__init__(config)

    @staticmethod
    def name() -> str:
        """Component name"""
        return "dummy_base_parameter_client_name"


# data for checking that the components have been initialized correctly
expected_count_keys = {
    "evaluator_episodes",
    "evaluator_steps",
    "executor_episodes",
    "executor_steps",
    "trainer_steps",
    "trainer_walltime",
}

initial_count_parameters = {
    "trainer_steps": np.array(0, dtype=np.int32),
    "trainer_walltime": np.array(0.0, dtype=np.float32),
    "evaluator_steps": np.array(0, dtype=np.int32),
    "evaluator_episodes": np.array(0, dtype=np.int32),
    "executor_episodes": np.array(0, dtype=np.int32),
    "executor_steps": np.array(0, dtype=np.int32),
}

######################
# SINGLE NETWORK DATA
######################

expected_network_keys = {
    "networks-network_agent_0",
    "networks-network_agent_1",
    "networks-network_agent_2",
}

expected_keys = expected_count_keys.union(expected_network_keys)

initial_parameters = {
    "networks-network_agent_0": {"weights": 0, "biases": 0},
    "networks-network_agent_1": {"weights": 1, "biases": 1},
    "networks-network_agent_2": {"weights": 2, "biases": 2},
    "trainer_steps": np.array(0, dtype=np.int32),
    "trainer_walltime": np.array(0.0, dtype=np.float32),
    "evaluator_steps": np.array(0, dtype=np.int32),
    "evaluator_episodes": np.array(0, dtype=np.int32),
    "executor_episodes": np.array(0, dtype=np.int32),
    "executor_steps": np.array(0, dtype=np.int32),
}

########################
# SEPARATE NETWORK DATA
########################

expected_separate_network_keys = {
    "policy_networks-network_agent_0",
    "policy_networks-network_agent_1",
    "policy_networks-network_agent_2",
    "critic_networks-network_agent_0",
    "critic_networks-network_agent_1",
    "critic_networks-network_agent_2",
}

expected_separate_network_keys = expected_count_keys.union(
    expected_separate_network_keys
)

initial_separate_network_parameters = {
    "policy_networks-network_agent_0": {"weights": 0, "biases": 0},
    "policy_networks-network_agent_1": {"weights": 1, "biases": 1},
    "policy_networks-network_agent_2": {"weights": 2, "biases": 2},
    "critic_networks-network_agent_0": {"weights": 0, "biases": 0},
    "critic_networks-network_agent_1": {"weights": 1, "biases": 1},
    "critic_networks-network_agent_2": {"weights": 2, "biases": 2},
    "trainer_steps": np.array(0, dtype=np.int32),
    "trainer_walltime": np.array(0.0, dtype=np.float32),
    "evaluator_steps": np.array(0, dtype=np.int32),
    "evaluator_episodes": np.array(0, dtype=np.int32),
    "executor_episodes": np.array(0, dtype=np.int32),
    "executor_steps": np.array(0, dtype=np.int32),
}

#########################
# SINGLE NETWORK FIXTURES
#########################


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
        store=SimpleNamespace(
            get_parameters={"trainer_steps": np.array(0, dtype=np.int32)}
        ),
        components=[],
    )

    return builder


###########################
# SEPARATE NETWORK FIXTURES
###########################


@pytest.fixture
def mock_builder_with_parameter_client_separate_networks() -> Builder:
    """Create a mock builder for testing.

    Has a parameter server and is designed for testing separate
    networks components.
    """

    builder = Builder(components=[])

    builder.store.networks = {
        "networks": {
            "network_agent_0": SimpleNamespace(
                policy_params={"weights": 0, "biases": 0},
                critic_params={"weights": 0, "biases": 0},
            ),
            "network_agent_1": SimpleNamespace(
                policy_params={"weights": 1, "biases": 1},
                critic_params={"weights": 1, "biases": 1},
            ),
            "network_agent_2": SimpleNamespace(
                policy_params={"weights": 2, "biases": 2},
                critic_params={"weights": 2, "biases": 2},
            ),
        }
    }

    builder.store.trainer_networks = {
        "trainer_0": ["network_agent_0", "network_agent_1", "network_agent_2"]
    }
    builder.store.trainer_id = "trainer_0"

    builder.store.parameter_server_client = ParameterServer(
        store=SimpleNamespace(
            get_parameters={"trainer_steps": np.array(0, dtype=np.int32)}
        ),
        components=[],
    )

    return builder


###############
# SHARED TESTS
###############


def test_base_parameter_client() -> None:
    """Test that count parameters are create in base \
        parameter client"""

    mock_client = MockBaseParameterClient(config=SimpleNamespace())

    keys, params = mock_client._set_up_count_parameters(params={})

    assert all([key in expected_keys for key in keys])

    assert params == initial_count_parameters


######################
# SINGLE NETWORK TESTS
######################


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

    assert all(
        [
            key in expected_keys
            for key in mock_builder.store.executor_parameter_client._all_keys
        ]
    )
    assert all(
        [
            key in expected_keys
            for key in mock_builder.store.executor_parameter_client._get_keys
        ]
    )

    assert mock_builder.store.executor_parameter_client._set_keys == [
        "executor_episodes",
        "executor_steps",
    ]
    assert (
        mock_builder.store.executor_parameter_client._parameters == initial_parameters
    )
    assert mock_builder.store.executor_parameter_client._get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._update_period == 100
    assert isinstance(
        mock_builder.store.executor_parameter_client._client, ParameterServer
    )

    assert mock_builder.store.executor_counts == initial_count_parameters


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

    assert all(
        [
            key in expected_keys
            for key in mock_builder.store.executor_parameter_client._all_keys
        ]
    )
    assert all(
        [
            key in expected_keys
            for key in mock_builder.store.executor_parameter_client._get_keys
        ]
    )

    assert mock_builder.store.executor_parameter_client._set_keys == [
        "evaluator_steps",
        "evaluator_episodes",
    ]
    assert (
        mock_builder.store.executor_parameter_client._parameters == initial_parameters
    )
    assert mock_builder.store.executor_parameter_client._get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._update_period == 100
    assert isinstance(
        mock_builder.store.executor_parameter_client._client, ParameterServer
    )

    assert mock_builder.store.executor_counts == initial_count_parameters


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

    assert all(
        [
            key in expected_keys
            for key in mock_builder.store.trainer_parameter_client._all_keys
        ]
    )
    assert all(
        [
            key in expected_count_keys
            for key in mock_builder.store.trainer_parameter_client._get_keys
        ]
    )

    assert mock_builder.store.trainer_parameter_client._set_keys == [
        "networks-network_agent_0",
        "networks-network_agent_1",
        "networks-network_agent_2",
    ]
    assert mock_builder.store.trainer_parameter_client._parameters == initial_parameters
    assert mock_builder.store.trainer_parameter_client._get_call_counter == 0
    assert mock_builder.store.trainer_parameter_client._set_call_counter == 0
    assert mock_builder.store.trainer_parameter_client._set_get_call_counter == 0
    assert mock_builder.store.trainer_parameter_client._update_period == 1
    assert isinstance(
        mock_builder.store.trainer_parameter_client._client, ParameterServer
    )

    assert mock_builder.store.trainer_counts == initial_count_parameters


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


########################
# SEPARATE NETWORK TESTS
########################


def test_executor_parameter_client_no_evaluator_with_parameter_client_separate_networks(
    mock_builder_with_parameter_client_separate_networks: Builder,
) -> None:
    """Test executor parameter client.

    Args:
        mock_builder_with_parameter_client_separate_networks : mava builder object
    """

    mock_builder = mock_builder_with_parameter_client_separate_networks
    mock_builder.store.is_evaluator = False
    exec_param_client = ExecutorParameterClientSeparateNetworks(
        config=ExecutorParameterClientConfig(executor_parameter_update_period=100)
    )
    exec_param_client.on_building_executor_parameter_client(builder=mock_builder)

    assert all(
        [
            key in expected_separate_network_keys
            for key in mock_builder.store.executor_parameter_client._all_keys
        ]
    )
    assert all(
        [
            key in expected_separate_network_keys
            for key in mock_builder.store.executor_parameter_client._get_keys
        ]
    )

    assert mock_builder.store.executor_parameter_client._set_keys == [
        "executor_episodes",
        "executor_steps",
    ]
    assert (
        mock_builder.store.executor_parameter_client._parameters
        == initial_separate_network_parameters
    )
    assert mock_builder.store.executor_parameter_client._get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._update_period == 100
    assert isinstance(
        mock_builder.store.executor_parameter_client._client, ParameterServer
    )

    assert mock_builder.store.executor_counts == initial_count_parameters


def test_executor_parameter_client_evaluator_with_parameter_client_separate_networks(
    mock_builder_with_parameter_client_separate_networks: Builder,
) -> None:
    """Test evaluator parameter client.

    Args:
        mock_builder_with_parameter_client_separate_networks: mava builder object
    """

    mock_builder = mock_builder_with_parameter_client_separate_networks
    mock_builder.store.is_evaluator = True
    exec_param_client = ExecutorParameterClientSeparateNetworks(
        config=ExecutorParameterClientConfig(executor_parameter_update_period=100)
    )
    exec_param_client.on_building_executor_parameter_client(builder=mock_builder)

    assert hasattr(mock_builder.store, "executor_parameter_client")

    assert all(
        [
            key in expected_separate_network_keys
            for key in mock_builder.store.executor_parameter_client._all_keys
        ]
    )
    assert all(
        [
            key in expected_separate_network_keys
            for key in mock_builder.store.executor_parameter_client._get_keys
        ]
    )

    assert mock_builder.store.executor_parameter_client._set_keys == [
        "evaluator_steps",
        "evaluator_episodes",
    ]
    assert (
        mock_builder.store.executor_parameter_client._parameters
        == initial_separate_network_parameters
    )
    assert mock_builder.store.executor_parameter_client._get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._update_period == 100
    assert isinstance(
        mock_builder.store.executor_parameter_client._client, ParameterServer
    )

    assert mock_builder.store.executor_counts == initial_count_parameters


def test_executor_parameter_client_with_no_parameter_client_separate_networks(
    mock_builder_with_parameter_client_separate_networks: Builder,
) -> None:
    """Test executor parameter server client when no parameter \
        server client is added to the builder"""

    mock_builder = mock_builder_with_parameter_client_separate_networks
    mock_builder.store.is_evaluator = True
    mock_builder.store.parameter_server_client = False
    exec_param_client = ExecutorParameterClientSeparateNetworks(
        config=ExecutorParameterClientConfig(executor_parameter_update_period=100)
    )

    exec_param_client.on_building_executor_parameter_client(mock_builder)

    assert mock_builder.store.executor_parameter_client is None


def test_trainer_parameter_client_separate_networks(
    mock_builder_with_parameter_client_separate_networks: Builder,
) -> None:
    """Test trainer parameter client.

    Args:
        mock_builder_with_parameter_client_separate_networks: mava builder object
    """

    mock_builder = mock_builder_with_parameter_client_separate_networks
    trainer_param_client = TrainerParameterClientSeparateNetworks()
    trainer_param_client.on_building_trainer_parameter_client(mock_builder)

    assert all(
        [
            key in expected_separate_network_keys
            for key in mock_builder.store.trainer_parameter_client._all_keys
        ]
    )
    assert all(
        [
            key in expected_count_keys
            for key in mock_builder.store.trainer_parameter_client._get_keys
        ]
    )

    assert mock_builder.store.trainer_parameter_client._set_keys == [
        "policy_networks-network_agent_0",
        "critic_networks-network_agent_0",
        "policy_networks-network_agent_1",
        "critic_networks-network_agent_1",
        "policy_networks-network_agent_2",
        "critic_networks-network_agent_2",
    ]
    assert (
        mock_builder.store.trainer_parameter_client._parameters
        == initial_separate_network_parameters
    )
    assert mock_builder.store.trainer_parameter_client._get_call_counter == 0
    assert mock_builder.store.trainer_parameter_client._set_call_counter == 0
    assert mock_builder.store.trainer_parameter_client._set_get_call_counter == 0
    assert mock_builder.store.trainer_parameter_client._update_period == 1
    assert isinstance(
        mock_builder.store.trainer_parameter_client._client, ParameterServer
    )

    assert mock_builder.store.trainer_counts == initial_count_parameters


def test_trainer_parameter_client_with_no_parameter_client_separate_networks(
    mock_builder_with_parameter_client_separate_networks: Builder,
) -> None:
    """Test trainer parameter server client when no parameter \
        server client is added to the builder"""

    mock_builder = mock_builder_with_parameter_client_separate_networks
    mock_builder.store.parameter_server_client = False
    trainer_param_client = TrainerParameterClientSeparateNetworks()

    trainer_param_client.on_building_trainer_parameter_client(mock_builder)

    assert mock_builder.store.trainer_parameter_client is None
