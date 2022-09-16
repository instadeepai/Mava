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

"""Tests for parameter server class for Jax-based Mava systems"""

import time
from types import SimpleNamespace
from typing import List

import pytest

from mava.callbacks import Callback
from mava.systems import ParameterServer
from tests.jax.hook_order_tracking import HookOrderTracking


class TestParameterServer(HookOrderTracking, ParameterServer):
    __test__ = False

    def __init__(
        self,
        store: SimpleNamespace,
        components: List[Callback],
    ) -> None:
        """Initialise the parameter server."""
        self.reset_hook_list()

        super().__init__(store, components)


@pytest.fixture
def test_parameter_server() -> ParameterServer:
    """Dummy parameter server with no components"""
    return TestParameterServer(
        store=SimpleNamespace(
            store_key="expected_value",
            global_config=SimpleNamespace(non_blocking_sleep_seconds=1),
            get_parameters="parameter_list",
        ),
        components=[],
    )


def test_store_loaded(test_parameter_server: TestParameterServer) -> None:
    """Test that store is loaded during init"""
    assert test_parameter_server.store.store_key == "expected_value"


def test_get_parameters_store(test_parameter_server: TestParameterServer) -> None:
    """Test that store is handled properly in get_parameters"""
    assert test_parameter_server.get_parameters("parameter_names") == "parameter_list"
    assert test_parameter_server.store._param_names == "parameter_names"


def test_set_parameters_store(test_parameter_server: TestParameterServer) -> None:
    """Test that store is handled properly in set_parameters"""
    set_params = {"parameter_name": "value"}
    test_parameter_server.set_parameters(set_params)
    assert test_parameter_server.store._set_params["parameter_name"] == "value"


def test_add_to_parameters_store(test_parameter_server: TestParameterServer) -> None:
    """Test that store is handled properly in add_to_parameters"""
    add_to_params = {"parameter_name": "value"}
    test_parameter_server.add_to_parameters(add_to_params)
    assert test_parameter_server.store._add_to_params["parameter_name"] == "value"


def test_step_sleep(test_parameter_server: TestParameterServer) -> None:
    """Test that step sleeps"""
    start = time.time()
    test_parameter_server.step()
    assert time.time() - start >= 1


def test_init_hook_order(test_parameter_server: TestParameterServer) -> None:
    """Test if init hooks are called in the correct order"""
    assert test_parameter_server.hook_list == [
        "on_parameter_server_init_start",
        "on_parameter_server_init",
        "on_parameter_server_init_checkpointer",
        "on_parameter_server_init_end",
    ]


def test_get_parameters_hook_order(test_parameter_server: TestParameterServer) -> None:
    """Test if get_parameters hooks are called in the correct order"""
    test_parameter_server.reset_hook_list()
    test_parameter_server.get_parameters("")
    assert test_parameter_server.hook_list == [
        "on_parameter_server_get_parameters_start",
        "on_parameter_server_get_parameters",
        "on_parameter_server_get_parameters_end",
    ]


def test_set_parameters_hook_order(test_parameter_server: TestParameterServer) -> None:
    """Test if set_parameters hooks are called in the correct order"""
    test_parameter_server.reset_hook_list()
    test_parameter_server.set_parameters({})
    assert test_parameter_server.hook_list == [
        "on_parameter_server_set_parameters_start",
        "on_parameter_server_set_parameters",
        "on_parameter_server_set_parameters_end",
    ]


def test_add_to_parameters_hook_order(
    test_parameter_server: TestParameterServer,
) -> None:
    """Test if add_to_parameters hooks are called in the correct order"""
    test_parameter_server.reset_hook_list()
    test_parameter_server.add_to_parameters({})
    assert test_parameter_server.hook_list == [
        "on_parameter_server_add_to_parameters_start",
        "on_parameter_server_add_to_parameters",
        "on_parameter_server_add_to_parameters_end",
    ]


def test_step_hook_order(test_parameter_server: TestParameterServer) -> None:
    """Test if step hooks are called in the correct order"""
    test_parameter_server.reset_hook_list()
    test_parameter_server.step()
    assert test_parameter_server.hook_list == [
        "on_parameter_server_run_loop_start",
        "on_parameter_server_run_loop_checkpoint",
        "on_parameter_server_run_loop",
        "on_parameter_server_run_loop_termination",
        "on_parameter_server_run_loop_end",
    ]
