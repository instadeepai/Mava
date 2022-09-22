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
from optax import EmptyState

from mava import constants
from mava.components.building.parameter_client import (
    BaseParameterClient,
    ExecutorParameterClient,
    ExecutorParameterClientConfig,
    TrainerParameterClient,
    TrainerParameterClientConfig,
)
from mava.systems.builder import Builder
from mava.systems.parameter_server import ParameterServer


class MockBaseParameterClient(BaseParameterClient):
    """Mock for Parameter client"""

    def __init__(self, config: Any) -> None:
        """Initialize mock base parameter client class to test the \
            _set_up_count_parameters method in BaseParameterClient."""
        super().__init__(config)

    @staticmethod
    def name() -> str:
        """Component name"""
        return "dummy_base_parameter_client_name"


# Data for checking that the components have been initialized correctly
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

expected_network_keys = {
    "policy_networks-network_agent_0",
    "policy_networks-network_agent_1",
    "policy_networks-network_agent_2",
    "policy_opt_state-network_agent_0",
    "policy_opt_state-network_agent_1",
    "policy_opt_state-network_agent_2",
    "critic_networks-network_agent_0",
    "critic_networks-network_agent_1",
    "critic_networks-network_agent_2",
    "critic_opt_state-network_agent_0",
    "critic_opt_state-network_agent_1",
    "critic_opt_state-network_agent_2",
}

expected_keys = expected_count_keys.union(expected_network_keys)

initial_parameters_trainer = {
    "policy_networks-network_agent_0": {"weights": 0, "biases": 0},
    "policy_networks-network_agent_1": {"weights": 1, "biases": 1},
    "policy_networks-network_agent_2": {"weights": 2, "biases": 2},
    "critic_networks-network_agent_0": {"weights": 0, "biases": 0},
    "critic_networks-network_agent_1": {"weights": 1, "biases": 1},
    "critic_networks-network_agent_2": {"weights": 2, "biases": 2},
    "policy_opt_state-network_agent_0": {constants.OPT_STATE_DICT_KEY: EmptyState()},
    "policy_opt_state-network_agent_1": {constants.OPT_STATE_DICT_KEY: EmptyState()},
    "policy_opt_state-network_agent_2": {constants.OPT_STATE_DICT_KEY: EmptyState()},
    "critic_opt_state-network_agent_0": {constants.OPT_STATE_DICT_KEY: EmptyState()},
    "critic_opt_state-network_agent_1": {constants.OPT_STATE_DICT_KEY: EmptyState()},
    "critic_opt_state-network_agent_2": {constants.OPT_STATE_DICT_KEY: EmptyState()},
    "trainer_steps": np.array(0, dtype=np.int32),
    "trainer_walltime": np.array(0.0, dtype=np.float32),
    "evaluator_steps": np.array(0, dtype=np.int32),
    "evaluator_episodes": np.array(0, dtype=np.int32),
    "executor_episodes": np.array(0, dtype=np.int32),
    "executor_steps": np.array(0, dtype=np.int32),
}

# Executor parameter client prameters does not include opt states
initial_parameters_executor = {
    k: v for k, v in initial_parameters_trainer.items() if "opt_state" not in k
}


@pytest.fixture
def mock_builder_with_parameter_client() -> Builder:
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

    builder.store.policy_opt_states = {}
    for net_key in builder.store.networks["networks"].keys():
        builder.store.policy_opt_states[net_key] = {
            constants.OPT_STATE_DICT_KEY: EmptyState()
        }
    builder.store.critic_opt_states = {}
    for net_key in builder.store.networks["networks"].keys():
        builder.store.critic_opt_states[net_key] = {
            constants.OPT_STATE_DICT_KEY: EmptyState()
        }

    builder.store.parameter_server_client = ParameterServer(
        store=SimpleNamespace(
            get_parameters={"trainer_steps": np.array(0, dtype=np.int32)}
        ),
        components=[],
    )

    return builder


def test_base_parameter_client() -> None:
    """Test that count parameters are create in base \
        parameter client"""

    mock_client = MockBaseParameterClient(config=SimpleNamespace())

    keys, params = mock_client._set_up_count_parameters(params={})

    assert all([key in expected_keys for key in keys])

    assert params == initial_count_parameters


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
        config=ExecutorParameterClientConfig(executor_parameter_update_period=500)
    )
    exec_param_client.on_building_executor_parameter_client(builder=mock_builder)

    # Ensure that set_keys and get_keys have no common elements
    assert (
        len(
            set(mock_builder.store.executor_parameter_client._get_keys)
            & set(mock_builder.store.executor_parameter_client._set_keys)
        )
        == 0
    )

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
        mock_builder.store.executor_parameter_client._parameters
        == initial_parameters_executor
    )
    assert mock_builder.store.executor_parameter_client._get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._update_period == 500
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
        config=ExecutorParameterClientConfig(executor_parameter_update_period=500)
    )
    exec_param_client.on_building_executor_parameter_client(builder=mock_builder)

    # Ensure that set_keys and get_keys have no common elements
    assert (
        len(
            set(mock_builder.store.executor_parameter_client._get_keys)
            & set(mock_builder.store.executor_parameter_client._set_keys)
        )
        == 0
    )

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
        mock_builder.store.executor_parameter_client._parameters
        == initial_parameters_executor
    )
    assert mock_builder.store.executor_parameter_client._get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_call_counter == 0
    assert mock_builder.store.executor_parameter_client._set_get_call_counter == 0
    assert mock_builder.store.executor_parameter_client._update_period == 500
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
    trainer_param_client = TrainerParameterClient(
        config=TrainerParameterClientConfig(trainer_parameter_update_period=500)
    )
    trainer_param_client.on_building_trainer_parameter_client(mock_builder)

    # Ensure that set_keys and get_keys have no common elements
    assert (
        len(
            set(mock_builder.store.trainer_parameter_client._get_keys)
            & set(mock_builder.store.trainer_parameter_client._set_keys)
        )
        == 0
    )
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
        "policy_networks-network_agent_0",
        "critic_networks-network_agent_0",
        "policy_opt_state-network_agent_0",
        "critic_opt_state-network_agent_0",
        "policy_networks-network_agent_1",
        "critic_networks-network_agent_1",
        "policy_opt_state-network_agent_1",
        "critic_opt_state-network_agent_1",
        "policy_networks-network_agent_2",
        "critic_networks-network_agent_2",
        "policy_opt_state-network_agent_2",
        "critic_opt_state-network_agent_2",
    ]
    assert (
        mock_builder.store.trainer_parameter_client._parameters
        == initial_parameters_trainer
    )
    assert mock_builder.store.trainer_parameter_client._get_call_counter == 0
    assert mock_builder.store.trainer_parameter_client._set_call_counter == 0
    assert mock_builder.store.trainer_parameter_client._set_get_call_counter == 0
    assert mock_builder.store.trainer_parameter_client._update_period == 500
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
