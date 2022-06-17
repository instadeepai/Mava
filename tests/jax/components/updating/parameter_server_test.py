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

    return mock_system_parameter_server


@pytest.fixture
def test_default_parameter_server() -> DefaultParameterServer:
    config = ParameterServerConfig(non_blocking_sleep_seconds=15)
    return DefaultParameterServer(config)


def test_on_parameter_server_init_start_parameter_creation(
    test_default_parameter_server: DefaultParameterServer, server: SystemParameterServer
) -> None:
    test_default_parameter_server.on_parameter_server_init_start(server)

    # Sleep seconds loaded into store for access in core parameter server
    assert server.store.non_blocking_sleep_seconds == 15

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
    assert server.store.parameters['net_type_1-agent_net_1'] == 'net_1_1_params'
    assert server.store.parameters['net_type_1-agent_net_2'] == 'net_1_2_params'
    assert server.store.parameters['net_type_2-agent_net_1'] == 'net_2_1_params'
    assert server.store.parameters['net_type_2-agent_net_2'] == 'net_2_2_params'


def test_on_parameter_server_init_start_no_checkpointer(
    test_default_parameter_server: DefaultParameterServer, server: SystemParameterServer
) -> None:
    test_default_parameter_server.config.checkpoint = False
    test_default_parameter_server.on_parameter_server_init_start(server)

    assert not hasattr(server.store, 'system_checkpointer')


def test_on_parameter_server_init_start_create_checkpointer(
    test_default_parameter_server: DefaultParameterServer, server: SystemParameterServer
) -> None:
    test_default_parameter_server.config.checkpoint = True
    test_default_parameter_server.on_parameter_server_init_start(server)

    assert server.store.last_checkpoint_time == 0
    assert server.store.checkpoint_minute_interval == ParameterServerConfig.checkpoint_minute_interval

    assert hasattr(server.store, 'system_checkpointer')
    # Test nothing more for now, since it's weird that a tf checkpointer is being used

