import time
from types import SimpleNamespace
from typing import Any, Dict, Sequence, Union

import numpy as np
import pytest

from mava.components.jax.updating.parameter_server import (
    DefaultParameterServer,
    ParameterServerConfig,
)
from mava.core_jax import SystemParameterServer


class MockSystemParameterServer(SystemParameterServer):
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


###########################
# SEPARATE NETWORK FIXTURES
###########################


@pytest.fixture
def server_separate_networks() -> SystemParameterServer:
    """Pytest fixture for mock system parameter server"""
    mock_system_parameter_server = MockSystemParameterServer()

    mock_system_parameter_server.store.network_factory = lambda: {
        "net_type_1": {
            "agent_net_1": SimpleNamespace(
                policy_params="net_1_1_params", critic_params="net_1_1_params"
            ),
            "agent_net_2": SimpleNamespace(
                policy_params="net_1_2_params", critic_params="net_1_2_params"
            ),
        },
        "net_type_2": {
            "agent_net_1": SimpleNamespace(
                policy_params="net_2_1_params", critic_params="net_2_1_params"
            ),
            "agent_net_2": SimpleNamespace(
                policy_params="net_2_2_params", critic_params="net_2_2_params"
            ),
        },
    }

    mock_system_parameter_server.store.parameters = {
        "param1": "param1_value",
        "param2": "param2_value",
        "param3": "param3_value",
    }

    return mock_system_parameter_server


@pytest.fixture
def test_default_parameter_server_separate_networks() -> DefaultParameterServer:  # noqa: E501
    """Pytest fixture for default parameter server"""
    config = ParameterServerConfig(non_blocking_sleep_seconds=15)
    return DefaultParameterServer(config)


########################
# SEPARATE NETWORK TESTS
########################


def test_on_parameter_server_init_start_parameter_creation_separate_networks(
    test_default_parameter_server_separate_networks: DefaultParameterServer,
    server_separate_networks: SystemParameterServer,
) -> None:
    """Test that parameters are correctly assigned to the store"""

    test_default_parameter_server = test_default_parameter_server_separate_networks
    server = server_separate_networks

    # Delete existing parameters from store, since following method will create them
    delattr(server.store, "parameters")
    test_default_parameter_server.on_parameter_server_init_start(server)

    # Parameters attribute in store
    assert hasattr(server.store, "parameters")

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
        assert server.store.parameters[required_int_key] == np.zeros(1, dtype=np.int32)
    for required_float_key in required_float_keys:
        assert server.store.parameters[required_float_key] == np.zeros(
            1, dtype=np.float32
        )

    # Parameter store network parameters
    assert server.store.parameters["policy_net_type_1-agent_net_1"] == "net_1_1_params"
    assert server.store.parameters["policy_net_type_1-agent_net_2"] == "net_1_2_params"
    assert server.store.parameters["policy_net_type_2-agent_net_1"] == "net_2_1_params"
    assert server.store.parameters["policy_net_type_2-agent_net_2"] == "net_2_2_params"
    assert server.store.parameters["critic_net_type_1-agent_net_1"] == "net_1_1_params"
    assert server.store.parameters["critic_net_type_1-agent_net_2"] == "net_1_2_params"
    assert server.store.parameters["critic_net_type_2-agent_net_1"] == "net_2_1_params"
    assert server.store.parameters["critic_net_type_2-agent_net_2"] == "net_2_2_params"


def test_on_parameter_server_init_start_no_checkpointer_separate_networks(
    test_default_parameter_server_separate_networks: DefaultParameterServer,
    server_separate_networks: SystemParameterServer,
) -> None:
    """Test init when no checkpointing specified"""

    test_default_parameter_server = test_default_parameter_server_separate_networks
    server = server_separate_networks

    test_default_parameter_server.config.checkpoint = False
    test_default_parameter_server.on_parameter_server_init_start(server)

    assert not hasattr(server.store, "system_checkpointer")


def test_on_parameter_server_init_start_create_checkpointer_separate_networks(
    test_default_parameter_server_separate_networks: DefaultParameterServer,
    server_separate_networks: SystemParameterServer,
) -> None:
    """Test init when checkpointer should be created"""

    test_default_parameter_server = test_default_parameter_server_separate_networks
    server = server_separate_networks

    test_default_parameter_server.config.checkpoint = True
    test_default_parameter_server.on_parameter_server_init_start(server)

    assert server.store.last_checkpoint_time == 0
    assert hasattr(server.store, "system_checkpointer")
    # Test nothing more for now, since it's weird that a tf checkpointer is being used


def test_on_parameter_server_get_parameters_single_separate_networks(
    test_default_parameter_server_separate_networks: DefaultParameterServer,
    server_separate_networks: SystemParameterServer,
) -> None:
    """Test get_parameters when only a single parameter is requested"""

    test_default_parameter_server = test_default_parameter_server_separate_networks
    server = server_separate_networks

    server.store._param_names = "param2"

    test_default_parameter_server.on_parameter_server_get_parameters(server)

    assert server.store.get_parameters == "param2_value"


def test_on_parameter_server_get_parameters_list_separate_networks(
    test_default_parameter_server_separate_networks: DefaultParameterServer,
    server_separate_networks: SystemParameterServer,
) -> None:
    """Test get_parameters when a list of parameters are requested"""

    test_default_parameter_server = test_default_parameter_server_separate_networks
    server = server_separate_networks

    server.store._param_names = ["param1", "param3"]

    test_default_parameter_server.on_parameter_server_get_parameters(server)

    assert server.store.get_parameters["param1"] == "param1_value"
    assert server.store.get_parameters["param3"] == "param3_value"
    assert "param2" not in server.store.get_parameters.keys()


def test_on_parameter_server_set_parameters_separate_networks(
    test_default_parameter_server_separate_networks: DefaultParameterServer,
    server_separate_networks: SystemParameterServer,
) -> None:
    """Test setting parameters"""

    test_default_parameter_server = test_default_parameter_server_separate_networks
    server = server_separate_networks

    server.store._set_params = {
        "param1": "param1_new_value",
        "param3": "param3_new_value",
    }

    test_default_parameter_server.on_parameter_server_set_parameters(server)

    assert server.store.parameters["param1"] == "param1_new_value"
    assert server.store.parameters["param2"] == "param2_value"
    assert server.store.parameters["param3"] == "param3_new_value"


def test_on_parameter_server_add_to_parameters_separate_networks(
    test_default_parameter_server_separate_networks: DefaultParameterServer,
    server_separate_networks: SystemParameterServer,
) -> None:
    """Test addition on parameters"""

    test_default_parameter_server = test_default_parameter_server_separate_networks
    server = server_separate_networks

    server.store.parameters["param3"] = 4
    server.store._add_to_params = {
        "param1": "_param1_add",
        "param3": 2,
    }

    test_default_parameter_server.on_parameter_server_add_to_parameters(server)

    assert server.store.parameters["param1"] == "param1_value_param1_add"
    assert server.store.parameters["param2"] == "param2_value"
    assert server.store.parameters["param3"] == 6


def test_on_parameter_server_run_loop_separate_networks(
    test_default_parameter_server_separate_networks: DefaultParameterServer,
    server_separate_networks: SystemParameterServer,
) -> None:
    """Test checkpointing in run loop"""

    test_default_parameter_server = test_default_parameter_server_separate_networks
    server = server_separate_networks

    server.store.last_checkpoint_time = 0

    # Do nothing if no checkpointer, even if checkpoint time has been reached
    test_default_parameter_server.config.checkpoint = False
    # Assert that the time to checkpoint has been reached
    assert (
        server.store.last_checkpoint_time
        + test_default_parameter_server.config.checkpoint_minute_interval * 60
        + 1
        < time.time()
    )
    test_default_parameter_server.on_parameter_server_run_loop(server)
    assert server.store.last_checkpoint_time == 0

    class DummyCheckpointer:
        def __init__(self) -> None:
            self.called = False

        def save(self) -> None:
            self.called = True

    server.store.system_checkpointer = DummyCheckpointer()
    test_default_parameter_server.config.checkpoint = True

    # Checkpoint if past time
    test_default_parameter_server.on_parameter_server_run_loop(server)
    assert server.store.system_checkpointer.called

    # Wait for next checkpoint
    server.store.system_checkpointer.called = False
    test_default_parameter_server.on_parameter_server_run_loop(server)
    assert not server.store.system_checkpointer.called
