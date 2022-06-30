from types import SimpleNamespace
from typing import List

import pytest

from mava.callbacks import Callback
from mava.systems.jax import Builder, ParameterServer
from tests.jax.hook_order_tracking import HookOrderTracking


class TestBuilder(HookOrderTracking, Builder):
    def __init__(
        self,
        components: List[Callback],
        global_config: SimpleNamespace,
    ) -> None:
        """Initialise the builder."""
        self.reset_hook_list()

        super().__init__(components=components, global_config=global_config)
        self.store.data_tables = ["data_table_1", "data_table_2"]


@pytest.fixture
def test_builder() -> Builder:
    """Dummy builder with no components."""
    return TestBuilder(
        components=[],
        global_config=SimpleNamespace(config_key="config_value"),
    )


def test_global_config_loaded(test_builder: TestBuilder) -> None:
    """Test that global config is loaded into the store during init()."""
    assert test_builder.store.global_config.config_key == "config_value"


def test_data_server_store(test_builder: TestBuilder) -> None:
    """Test that store is handled correctly in data_server()."""
    assert test_builder.data_server() == ["data_table_1", "data_table_2"]


def test_parameter_server_store(test_builder: TestBuilder) -> None:
    """Test that store is handled correctly in parameter_server()."""
    parameter_server = test_builder.parameter_server()
    assert isinstance(parameter_server, ParameterServer)
    assert parameter_server.store == test_builder.store
