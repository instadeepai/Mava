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

"""Parameter server unit test"""

from types import SimpleNamespace
from typing import Any, Dict, Sequence, Union

import numpy as np
import pytest
from optax import EmptyState

from mava import constants
from mava.components.updating.parameter_server import (
    DefaultParameterServer,
    ParameterServerConfig,
)
from mava.core_jax import SystemParameterServer


class MockSystemParameterServer(SystemParameterServer):
    """Mock for paramter server"""

    def get_parameters(
        self, names: Union[str, Sequence[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Get parameters from the parameter server.

        Args:
            names : Names of the parameters to get
        Returns:
            The parameters that were requested
        """
        return {}

    def set_parameters(self, set_params: Dict[str, Any]) -> None:
        """Set parameters in the parameter server.

        Args:
            set_params : The values to set the parameters to
        """
        pass

    def add_to_parameters(self, add_to_params: Dict[str, Any]) -> None:
        """Add to the parameters in the parameter server.

        Args:
            add_to_params : values to add to the parameters
        """
        pass

    def run(self) -> None:
        """Run the parameter server. This function allows for checkpointing and other \
        centralised computations to be performed by the parameter server."""
        pass


@pytest.fixture
def mock_system_parameter_server() -> SystemParameterServer:
    """Pytest fixture for mock system parameter server"""
    mock_system_parameter_server = MockSystemParameterServer()
    mock_system_parameter_server.store.network_factory = lambda: {
        "agent_net_1": SimpleNamespace(
            policy_params="net_1_1_params", critic_params="net_1_1_params"
        ),
        "agent_net_2": SimpleNamespace(
            policy_params="net_1_2_params", critic_params="net_1_2_params"
        ),
    }
    mock_system_parameter_server.store.networks = (
        mock_system_parameter_server.store.network_factory()
    )
    mock_system_parameter_server.store.policy_opt_states = {}
    mock_system_parameter_server.store.critic_opt_states = {}
    for net_key in mock_system_parameter_server.store.networks.keys():
        mock_system_parameter_server.store.policy_opt_states[net_key] = {
            constants.OPT_STATE_DICT_KEY: EmptyState()
        }
        mock_system_parameter_server.store.critic_opt_states[net_key] = {
            constants.OPT_STATE_DICT_KEY: EmptyState()
        }

    mock_system_parameter_server.store.norm_params = {}
    mock_system_parameter_server.store.norm_params[
        constants.OBS_NORM_STATE_DICT_KEY
    ] = EmptyState()
    mock_system_parameter_server.store.norm_params[
        constants.VALUES_NORM_STATE_DICT_KEY
    ] = EmptyState()

    mock_system_parameter_server.store.parameters = {
        "param1": "param1_value",
        "param2": "param2_value",
        "param3": "param3_value",
        "evaluator_or_trainer_failed": False,
        "num_executor_failed": 0,
    }

    mock_system_parameter_server.store.metrics_checkpoint = ["mean_episode_return"]

    mock_system_parameter_server.store.num_executors = 2

    mock_system_parameter_server.store.checkpoint_best_perf = True

    return mock_system_parameter_server


@pytest.fixture
def test_default_parameter_server() -> DefaultParameterServer:  # noqa: E501
    """Pytest fixture for default parameter server"""
    config = ParameterServerConfig(non_blocking_sleep_seconds=15)
    return DefaultParameterServer(config)


def test_on_parameter_server_init_start_parameter_creation(
    test_default_parameter_server: DefaultParameterServer,
    mock_system_parameter_server: SystemParameterServer,
) -> None:
    """Test that parameters are correctly assigned to the store"""

    # Delete existing parameters from store, since following method will create them
    delattr(mock_system_parameter_server.store, "parameters")
    test_default_parameter_server.on_parameter_server_init_start(
        mock_system_parameter_server
    )

    # Parameters attribute in store
    assert hasattr(mock_system_parameter_server.store, "parameters")
    assert hasattr(mock_system_parameter_server.store, "experiment_path")
    assert (
        mock_system_parameter_server.store.experiment_path
        == test_default_parameter_server.config.experiment_path
    )

    # Parameter store training / executing info
    required_int_keys = {
        "trainer_steps",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    }
    required_float_keys = {"trainer_walltime"}
    for required_int_key in required_int_keys:
        assert mock_system_parameter_server.store.parameters[
            required_int_key
        ] == np.zeros(1, dtype=np.int32)
    for required_float_key in required_float_keys:
        assert mock_system_parameter_server.store.parameters[
            required_float_key
        ] == np.zeros(1, dtype=np.float32)

    # Parameter store network parameters
    assert (
        mock_system_parameter_server.store.parameters["policy_network-agent_net_1"]
        == "net_1_1_params"
    )
    assert (
        mock_system_parameter_server.store.parameters["policy_network-agent_net_2"]
        == "net_1_2_params"
    )
    assert (
        mock_system_parameter_server.store.parameters["critic_network-agent_net_1"]
        == "net_1_1_params"
    )
    assert (
        mock_system_parameter_server.store.parameters["critic_network-agent_net_2"]
        == "net_1_2_params"
    )
    assert not mock_system_parameter_server.store.parameters[
        "evaluator_or_trainer_failed"
    ]
    assert mock_system_parameter_server.store.parameters["num_executor_failed"] == 0


def test_on_parameter_server_get_parameters_single(
    test_default_parameter_server: DefaultParameterServer,
    mock_system_parameter_server: SystemParameterServer,
) -> None:
    """Test get_parameters when only a single parameter is requested"""

    mock_system_parameter_server.store._param_names = "param2"

    test_default_parameter_server.on_parameter_server_get_parameters(
        mock_system_parameter_server
    )

    assert mock_system_parameter_server.store.get_parameters == "param2_value"


def test_on_parameter_server_get_parameters_list(
    test_default_parameter_server: DefaultParameterServer,
    mock_system_parameter_server: SystemParameterServer,
) -> None:
    """Test get_parameters when a list of parameters are requested"""

    mock_system_parameter_server.store._param_names = ["param1", "param3"]

    test_default_parameter_server.on_parameter_server_get_parameters(
        mock_system_parameter_server
    )

    assert mock_system_parameter_server.store.get_parameters["param1"] == "param1_value"
    assert mock_system_parameter_server.store.get_parameters["param3"] == "param3_value"
    assert "param2" not in mock_system_parameter_server.store.get_parameters.keys()


def test_on_mock_system_parameter_server_set_parameters(
    test_default_parameter_server: DefaultParameterServer,
    mock_system_parameter_server: SystemParameterServer,
) -> None:
    """Test setting parameters"""

    mock_system_parameter_server.store._set_params = {
        "param1": "param1_new_value",
        "param3": "param3_new_value",
    }

    test_default_parameter_server.on_parameter_server_set_parameters(
        mock_system_parameter_server
    )

    assert mock_system_parameter_server.store.parameters["param1"] == "param1_new_value"
    assert mock_system_parameter_server.store.parameters["param2"] == "param2_value"
    assert mock_system_parameter_server.store.parameters["param3"] == "param3_new_value"


def test_on_parameter_server_add_to_parameters(
    test_default_parameter_server: DefaultParameterServer,
    mock_system_parameter_server: SystemParameterServer,
) -> None:
    """Test addition on parameters"""

    mock_system_parameter_server.store.parameters["param3"] = 4
    mock_system_parameter_server.store._add_to_params = {
        "param1": "_param1_add",
        "param3": 2,
    }

    test_default_parameter_server.on_parameter_server_add_to_parameters(
        mock_system_parameter_server
    )

    assert (
        mock_system_parameter_server.store.parameters["param1"]
        == "param1_value_param1_add"
    )
    assert mock_system_parameter_server.store.parameters["param2"] == "param2_value"
    assert mock_system_parameter_server.store.parameters["param3"] == 6

    # Test that the number of num_executor_failed got incremneted
    mock_system_parameter_server.store._add_to_params = {"num_executor_failed": 1}

    test_default_parameter_server.on_parameter_server_add_to_parameters(
        mock_system_parameter_server
    )
    assert mock_system_parameter_server.store.parameters["num_executor_failed"] == 1
