from types import SimpleNamespace
from typing import List

import pytest

from mava.callbacks import Callback
from mava.systems.jax import Builder, Executor, ParameterServer
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
        self.store.system_executor = "system_executor"
        self.store.adder = "adder"


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


def test_executor_store_when_is_evaluator(test_builder: TestBuilder) -> None:
    """Test that store is handled correctly in executor() when it's an evaluator."""
    executor_id = "evaluator"
    data_server_client = "data_server_client"
    parameter_server_client = "parameter_server_client"
    assert (
        test_builder.executor(
            executor_id=executor_id,
            data_server_client=data_server_client,
            parameter_server_client=parameter_server_client,
        )
        == "system_executor"
    )

    assert test_builder.store.executor_id == executor_id
    assert test_builder.store.data_server_client == data_server_client
    assert test_builder.store.parameter_server_client == parameter_server_client
    assert test_builder.store.is_evaluator
    assert test_builder.store.adder is None

    assert isinstance(test_builder.store.executor, Executor)
    assert test_builder.store.executor.store == test_builder.store


def test_executor_store_when_executor(test_builder: TestBuilder) -> None:
    """Test that store is handled correctly in executor() when it's an executor."""
    executor_id = "executor"
    data_server_client = "data_server_client"
    parameter_server_client = "parameter_server_client"
    assert (
        test_builder.executor(
            executor_id=executor_id,
            data_server_client=data_server_client,
            parameter_server_client=parameter_server_client,
        )
        == "system_executor"
    )

    assert test_builder.store.executor_id == executor_id
    assert test_builder.store.data_server_client == data_server_client
    assert test_builder.store.parameter_server_client == parameter_server_client
    assert not test_builder.store.is_evaluator
    assert test_builder.store.adder == "adder"

    assert isinstance(test_builder.store.executor, Executor)
    assert test_builder.store.executor.store == test_builder.store
