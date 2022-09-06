from types import SimpleNamespace
from typing import Any, Dict, Sequence, Union

import numpy as np
import pytest

from mava.components.jax.updating.parameter_server import (
    DefaultParameterServer,
    ParameterServerConfig,
    ParameterServerSeparateNetworks,
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


#########################
# SINGLE NETWORK FIXTURES
#########################


@pytest.fixture
def server() -> SystemParameterServer:
    """Pytest fixture for mock system parameter server"""
    mock_system_parameter_server = MockSystemParameterServer()

    mock_system_parameter_server.store.network_factory = lambda: {
        "net_type_1": {
            "agent_net_1": SimpleNamespace(params="net_1_1_params"),
            "agent_net_2": SimpleNamespace(params="net_1_2_params"),
        },
        "net_type_2": {
            "agent_net_1": SimpleNamespace(params="net_2_1_params"),
            "agent_net_2": SimpleNamespace(params="net_2_2_params"),
        },
    }

    mock_system_parameter_server.store.parameters = {
        "param1": "param1_value",
        "param2": "param2_value",
        "param3": "param3_value",
    }

    return mock_system_parameter_server


@pytest.fixture
def test_default_parameter_server() -> DefaultParameterServer:
    """Pytest fixture for default parameter server"""
    config = ParameterServerConfig(non_blocking_sleep_seconds=15)
    return DefaultParameterServer(config)


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
def test_default_parameter_server_separate_networks() -> ParameterServerSeparateNetworks:  # noqa: E501
    """Pytest fixture for default parameter server"""
    config = ParameterServerConfig(non_blocking_sleep_seconds=15)
    return ParameterServerSeparateNetworks(config)


######################
# SINGLE NETWORK TESTS
######################


def test_on_parameter_server_init_start(
    test_default_parameter_server: DefaultParameterServer, server: SystemParameterServer
) -> None:
    """Test that parameters are correctly assigned to the store"""

    # Delete existing parameters from store, since following method will create them
    delattr(server.store, "parameters")
    test_default_parameter_server.on_parameter_server_init_start(server)

    # Parameters attribute in store
    assert hasattr(server.store, "parameters")
    assert hasattr(server.store, "experiment_path")
    assert (
        server.store.experiment_path
        == test_default_parameter_server.config.experiment_path
    )

    # Parameter store training / executing info
    required_int_keys = {
        "trainer_steps",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
        # "seed",
        # "optimizer_state",
    }
    required_float_keys = {"trainer_walltime"}
    for required_int_key in required_int_keys:
        assert server.store.parameters[required_int_key] == np.zeros(1, dtype=np.int32)
    for required_float_key in required_float_keys:
        assert server.store.parameters[required_float_key] == np.zeros(
            1, dtype=np.float32
        )

    # Parameter store network parameters
    assert server.store.parameters["net_type_1-agent_net_1"] == "net_1_1_params"
    assert server.store.parameters["net_type_1-agent_net_2"] == "net_1_2_params"
    assert server.store.parameters["net_type_2-agent_net_1"] == "net_2_1_params"
    assert server.store.parameters["net_type_2-agent_net_2"] == "net_2_2_params"

    assert hasattr(server.store, "saveable_parameters")
    assert all(
        not (type(var) == tuple and len(var) == 0)
        for var in server.store.saveable_parameters.values()
    )
    assert hasattr(server.store, "experiment_path")


def test_on_parameter_server_get_parameters_single(
    test_default_parameter_server: DefaultParameterServer, server: SystemParameterServer
) -> None:
    """Test get_parameters when only a single parameter is requested"""
    server.store._param_names = "param2"

    test_default_parameter_server.on_parameter_server_get_parameters(server)

    assert server.store.get_parameters == "param2_value"


def test_on_parameter_server_get_parameters_list(
    test_default_parameter_server: DefaultParameterServer, server: SystemParameterServer
) -> None:
    """Test get_parameters when a list of parameters are requested"""
    server.store._param_names = ["param1", "param3"]

    test_default_parameter_server.on_parameter_server_get_parameters(server)

    assert server.store.get_parameters["param1"] == "param1_value"
    assert server.store.get_parameters["param3"] == "param3_value"
    assert "param2" not in server.store.get_parameters.keys()


def test_on_parameter_server_set_parameters(
    test_default_parameter_server: DefaultParameterServer, server: SystemParameterServer
) -> None:
    """Test setting parameters"""
    server.store._set_params = {
        "param1": "param1_new_value",
        "param3": "param3_new_value",
    }

    test_default_parameter_server.on_parameter_server_set_parameters(server)

    assert server.store.parameters["param1"] == "param1_new_value"
    assert server.store.parameters["param2"] == "param2_value"
    assert server.store.parameters["param3"] == "param3_new_value"


def test_on_parameter_server_add_to_parameters(
    test_default_parameter_server: DefaultParameterServer, server: SystemParameterServer
) -> None:
    """Test addition on parameters"""
    server.store.parameters["param3"] = 4
    server.store._add_to_params = {
        "param1": "_param1_add",
        "param3": 2,
    }

    test_default_parameter_server.on_parameter_server_add_to_parameters(server)

    assert server.store.parameters["param1"] == "param1_value_param1_add"
    assert server.store.parameters["param2"] == "param2_value"
    assert server.store.parameters["param3"] == 6


########################
# SEPARATE NETWORK TESTS
########################


def test_on_parameter_server_init_start_separate_networks(
    test_default_parameter_server_separate_networks: ParameterServerSeparateNetworks,
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
    assert hasattr(server.store, "experiment_path")
    assert (
        server.store.experiment_path
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

    assert hasattr(server.store, "saveable_parameters")
    assert all(
        not (type(var) == tuple and len(var) == 0)
        for var in server.store.saveable_parameters.values()
    )
    assert hasattr(server.store, "experiment_path")


def test_on_parameter_server_get_parameters_single_separate_networks(
    test_default_parameter_server_separate_networks: ParameterServerSeparateNetworks,
    server_separate_networks: SystemParameterServer,
) -> None:
    """Test get_parameters when only a single parameter is requested"""

    test_default_parameter_server = test_default_parameter_server_separate_networks
    server = server_separate_networks

    server.store._param_names = "param2"

    test_default_parameter_server.on_parameter_server_get_parameters(server)

    assert server.store.get_parameters == "param2_value"


def test_on_parameter_server_get_parameters_list_separate_networks(
    test_default_parameter_server_separate_networks: ParameterServerSeparateNetworks,
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
    test_default_parameter_server_separate_networks: ParameterServerSeparateNetworks,
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
    test_default_parameter_server_separate_networks: ParameterServerSeparateNetworks,
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
