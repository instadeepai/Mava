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

"""Builder unit test"""

from types import SimpleNamespace
from typing import List

import pytest

from mava.callbacks import Callback
from mava.components.building import Logger
from mava.systems import Builder, Executor, ParameterServer, Trainer
from tests.jax.hook_order_tracking import HookOrderTracking


class TestBuilder(HookOrderTracking, Builder):
    """Mock for the builder"""

    __test__ = False

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

        self.store.data_key = [1234, 1234]
        self.store.eval_key = [1234, 1234]
        self.store.param_key = [1234, 1234]
        self.store.executor_keys = [[1234, 1234]]
        self.store.trainer_keys = [[1234, 1234]]


@pytest.fixture
def test_builder() -> Builder:
    """Dummy builder with no components."""
    return TestBuilder(
        components=[Logger()],
        global_config=SimpleNamespace(config_key="config_value"),
    )


def test_global_config_loaded(test_builder: TestBuilder) -> None:
    """Test that global config is loaded into the store during init()."""
    assert test_builder.store.global_config.config_key == "config_value"
    assert len(test_builder.callbacks) == 1
    assert isinstance(test_builder.callbacks[0], Logger)


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
    executor_id = "executor_0"
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


def test_trainer_store(test_builder: TestBuilder) -> None:
    """Test that store is handled correctly in trainer()."""
    trainer_id = "trainer_0"
    data_server_client = "data_server_client"
    parameter_server_client = "parameter_server_client"

    trainer = test_builder.trainer(
        trainer_id=trainer_id,
        data_server_client=data_server_client,
        parameter_server_client=parameter_server_client,
    )
    assert isinstance(trainer, Trainer)
    assert trainer.store == test_builder.store

    assert test_builder.store.trainer_id == trainer_id
    assert test_builder.store.data_server_client == data_server_client
    assert test_builder.store.parameter_server_client == parameter_server_client


def test_init_hook_order(test_builder: TestBuilder) -> None:
    """Test if init() hooks are called in the correct order."""
    assert test_builder.hook_list == [
        "on_building_init_start",
        "on_building_init",
        "on_building_init_end",
    ]


def test_data_server_hook_order(test_builder: TestBuilder) -> None:
    """Test if data_server() hooks are called in the correct order."""
    test_builder.reset_hook_list()
    test_builder.data_server()
    assert test_builder.hook_list == [
        "on_building_data_server_start",
        "on_building_data_server_adder_signature",
        "on_building_data_server_rate_limiter",
        "on_building_data_server",
        "on_building_data_server_end",
    ]


def test_parameter_server_hook_order(test_builder: TestBuilder) -> None:
    """Test if parameter_server() hooks are called in the correct order."""
    test_builder.reset_hook_list()
    test_builder.parameter_server()
    assert test_builder.hook_list == [
        "on_building_parameter_server_start",
        "on_building_parameter_server",
        "on_building_parameter_server_end",
    ]


def test_executor_hook_order_when_executor(test_builder: TestBuilder) -> None:
    """Test if executor() hooks are called in the correct order when executor."""
    test_builder.reset_hook_list()
    test_builder.executor(
        executor_id="executor_0", data_server_client="", parameter_server_client=""
    )
    assert test_builder.hook_list == [
        "on_building_executor_start",
        "on_building_executor_adder_priority",
        "on_building_executor_adder",
        "on_building_executor_logger",
        "on_building_executor_parameter_client",
        "on_building_executor",
        "on_building_executor_environment",
        "on_building_executor_environment_loop",
        "on_building_executor_end",
    ]


def test_executor_hook_order_when_evaluator(test_builder: TestBuilder) -> None:
    """Test if executor() hooks are called in the correct order when evaluator."""
    test_builder.reset_hook_list()
    test_builder.executor(
        executor_id="evaluator", data_server_client="", parameter_server_client=""
    )
    assert test_builder.hook_list == [
        "on_building_executor_start",
        "on_building_executor_logger",
        "on_building_executor_parameter_client",
        "on_building_executor",
        "on_building_executor_environment",
        "on_building_executor_environment_loop",
        "on_building_executor_end",
    ]


def test_trainer_hook_order(test_builder: TestBuilder) -> None:
    """Test if trainer() hooks are called in the correct order."""
    test_builder.reset_hook_list()
    test_builder.trainer(
        trainer_id="trainer_0", data_server_client="", parameter_server_client=""
    )
    assert test_builder.hook_list == [
        "on_building_trainer_start",
        "on_building_trainer_logger",
        "on_building_trainer_dataset",
        "on_building_trainer_parameter_client",
        "on_building_trainer",
        "on_building_trainer_end",
    ]


def test_build_hook_order(test_builder: TestBuilder) -> None:
    """Test if build() hooks are called in the correct order."""
    test_builder.reset_hook_list()
    test_builder.build()
    assert test_builder.hook_list == [
        "on_building_start",
        "on_building_program_nodes",
        "on_building_end",
    ]


def test_launch_hook_order(test_builder: TestBuilder) -> None:
    """Test if launch() hooks are called in the correct order."""
    test_builder.reset_hook_list()
    test_builder.launch()
    assert test_builder.hook_list == [
        "on_building_launch_start",
        "on_building_launch",
        "on_building_launch_end",
    ]
