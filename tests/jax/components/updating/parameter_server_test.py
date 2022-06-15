from types import SimpleNamespace
from typing import Any, Dict, Sequence, Union

import pytest

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
def mock_system_parameter_server() -> SystemParameterServer:
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
