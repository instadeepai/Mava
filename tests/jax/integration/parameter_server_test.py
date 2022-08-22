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

"""Tests for parameter server for Jax-based Mava systems"""
import os
import signal
import time
from typing import Any, Dict

import jax.numpy as jnp
import pytest
from acme.jax import savers

from mava.systems.jax import System
from tests.jax.integration.mock_systems import (
    mock_system_multi_process,
    mock_system_multi_thread,
    mock_system_single_process,
)


@pytest.fixture
def test_system_sp() -> System:
    """A single process built system"""
    return mock_system_single_process()


@pytest.fixture
def test_system_mp() -> System:
    """A multi process built system"""
    return mock_system_multi_process()


@pytest.fixture
def test_system_mt() -> System:
    """A multi thread built system"""
    return mock_system_multi_thread()


def test_parameter_server_single_process(test_system_sp: System) -> None:
    """Test if the paraameter server instantiates processes as expected."""
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system_sp._builder.store.system_build

    # System parameter server
    assert type(parameter_server.store.system_checkpointer) == savers.Checkpointer

    for network in parameter_server.store.parameters["networks-network_agent"].values():
        assert list(network.keys()) == ["w", "b"]
        assert jnp.size(network["w"]) != 0
        assert jnp.size(network["b"]) != 0

    param_without_net = parameter_server.store.parameters.copy()
    del param_without_net["networks-network_agent"]
    assert param_without_net == {
        "trainer_steps": jnp.zeros(1, dtype=jnp.int32),
        "trainer_walltime": jnp.zeros(1, dtype=jnp.float32),
        "evaluator_steps": jnp.zeros(1, dtype=jnp.int32),
        "evaluator_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_steps": jnp.zeros(1, dtype=jnp.int32),
    }

    ############################get_parameters test#####################################

    parameter_server.get_parameters("trainer_steps")
    assert parameter_server.store.get_parameters == jnp.zeros(1, dtype=jnp.int32)

    parameter_server.get_parameters("networks-network_agent")
    assert (
        parameter_server.store.get_parameters
        == parameter_server.store.parameters["networks-network_agent"]
    )

    # get multiple params
    parameter_server.get_parameters(["executor_episodes", "executor_steps"])
    assert parameter_server.store.get_parameters == {
        "executor_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_steps": jnp.zeros(1, dtype=jnp.int32),
    }

    ############################set_parameters test#####################################
    params = {
        "trainer_steps": jnp.zeros(2, dtype=jnp.int32),
        "trainer_walltime": jnp.zeros(2, dtype=jnp.float32),
        "evaluator_steps": jnp.zeros(2, dtype=jnp.int32),
        "evaluator_episodes": jnp.zeros(2, dtype=jnp.int32),
        "executor_episodes": jnp.zeros(2, dtype=jnp.int32),
        "executor_steps": jnp.zeros(2, dtype=jnp.int32),
    }

    parameter_server.set_parameters(params)
    list_param = [
        "trainer_steps",
        "trainer_walltime",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    ]
    for param in list_param:
        assert jnp.array_equal(
            parameter_server.store.parameters[param], jnp.array([0, 0])
        )

    # set non existing param
    param1: Dict[str, str] = {"wrong_param": "test"}
    with pytest.raises(AssertionError):
        assert parameter_server.set_parameters(param1)

    ############################add_to_parameters_test#####################################
    param2: Dict[str, Any] = {
        "trainer_steps": jnp.array([1, 3]),
    }
    parameter_server.add_to_parameters(param2)

    assert jnp.array_equal(
        parameter_server.store.parameters["trainer_steps"], jnp.array([1, 3])
    )  # [0,0]+[1,3]

    ###################################step_test##########################################
    # before running step
    assert not parameter_server.store.last_checkpoint_time
    assert parameter_server.store.system_checkpointer._last_saved == 0

    # run step function
    parameter_server.step()

    assert parameter_server.store.last_checkpoint_time
    assert parameter_server.store.last_checkpoint_time < time.time()
    assert parameter_server.store.system_checkpointer._last_saved != 0
    assert parameter_server.store.system_checkpointer._last_saved < time.time()


def test_parameter_server_multi_thread(test_system_mt: System) -> None:
    """Test if the paraameter server instantiates processes as expected."""
    (parameter_server_node,) = test_system_mt._builder.store.program._program._groups[
        "parameter_server"
    ]
    parameter_server_node.disable_run()

    test_system_mt.launch()
    parameter_server = parameter_server_node._construct_instance()

    # System parameter server
    assert type(parameter_server.store.system_checkpointer) == savers.Checkpointer

    for network in parameter_server.store.parameters["networks-network_agent"].values():
        assert list(network.keys()) == ["w", "b"]
        assert jnp.size(network["w"]) != 0
        assert jnp.size(network["b"]) != 0

    param_without_net = parameter_server.store.parameters.copy()
    del param_without_net["networks-network_agent"]
    assert param_without_net == {
        "trainer_steps": jnp.zeros(1, dtype=jnp.int32),
        "trainer_walltime": jnp.zeros(1, dtype=jnp.float32),
        "evaluator_steps": jnp.zeros(1, dtype=jnp.int32),
        "evaluator_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_steps": jnp.zeros(1, dtype=jnp.int32),
    }

    ############################get_parameters test#####################################

    parameter_server.get_parameters("trainer_steps")
    assert parameter_server.store.get_parameters == jnp.zeros(1, dtype=jnp.int32)

    parameter_server.get_parameters("networks-network_agent")
    assert (
        parameter_server.store.get_parameters
        == parameter_server.store.parameters["networks-network_agent"]
    )

    # get multiple params
    parameter_server.get_parameters(["executor_episodes", "executor_steps"])
    assert parameter_server.store.get_parameters == {
        "executor_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_steps": jnp.zeros(1, dtype=jnp.int32),
    }

    ############################set_parameters test#####################################
    params = {
        "trainer_steps": jnp.zeros(2, dtype=jnp.int32),
        "trainer_walltime": jnp.zeros(2, dtype=jnp.float32),
        "evaluator_steps": jnp.zeros(2, dtype=jnp.int32),
        "evaluator_episodes": jnp.zeros(2, dtype=jnp.int32),
        "executor_episodes": jnp.zeros(2, dtype=jnp.int32),
        "executor_steps": jnp.zeros(2, dtype=jnp.int32),
    }

    parameter_server.set_parameters(params)
    list_param = [
        "trainer_steps",
        "trainer_walltime",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    ]
    for param in list_param:
        assert jnp.array_equal(
            parameter_server.store.parameters[param], jnp.array([0, 0])
        )

    # set non existing param
    param1: Dict[str, str] = {"wrong_param": "test"}
    with pytest.raises(AssertionError):
        assert parameter_server.set_parameters(param1)

    ############################add_to_parameters_test#####################################
    param2: Dict[str, Any] = {
        "trainer_steps": jnp.array([1, 3]),
    }
    parameter_server.add_to_parameters(param2)

    assert jnp.array_equal(
        parameter_server.store.parameters["trainer_steps"], jnp.array([1, 3])
    )  # [0,0]+[1,3]

    ###################################step_test##########################################
    # before running step
    assert not parameter_server.store.last_checkpoint_time
    assert parameter_server.store.system_checkpointer._last_saved == 0
    # run step function
    parameter_server.store.global_config.non_blocking_sleep_seconds = 0
    parameter_server.step()

    assert parameter_server.store.last_checkpoint_time
    assert parameter_server.store.last_checkpoint_time < time.time()
    assert parameter_server.store.system_checkpointer._last_saved != 0
    assert parameter_server.store.system_checkpointer._last_saved < time.time()


def test_parameter_server_multi_process(test_system_mp: System) -> None:
    """Test if the paraameter server instantiates processes as expected."""

    (parameter_server_node,) = test_system_mp._builder.store.program._program._groups[
        "parameter_server"
    ]
    parameter_server_node.disable_run()

    # pid is necessary to stop the launcher once the test is done
    pid = os.getpid()

    test_system_mp.launch()

    parameter_server = parameter_server_node._construct_instance()

    # System parameter server
    assert type(parameter_server.store.system_checkpointer) == savers.Checkpointer

    for network in parameter_server.store.parameters["networks-network_agent"].values():
        assert list(network.keys()) == ["w", "b"]
        assert jnp.size(network["w"]) != 0
        assert jnp.size(network["b"]) != 0

    param_without_net = parameter_server.store.parameters.copy()
    del param_without_net["networks-network_agent"]
    assert param_without_net == {
        "trainer_steps": jnp.zeros(1, dtype=jnp.int32),
        "trainer_walltime": jnp.zeros(1, dtype=jnp.float32),
        "evaluator_steps": jnp.zeros(1, dtype=jnp.int32),
        "evaluator_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_steps": jnp.zeros(1, dtype=jnp.int32),
    }

    ############################get_parameters test#####################################

    parameter_server.get_parameters("trainer_steps")
    assert parameter_server.store.get_parameters == jnp.zeros(1, dtype=jnp.int32)

    parameter_server.get_parameters("networks-network_agent")
    assert (
        parameter_server.store.get_parameters
        == parameter_server.store.parameters["networks-network_agent"]
    )

    # get multiple params
    parameter_server.get_parameters(["executor_episodes", "executor_steps"])
    assert parameter_server.store.get_parameters == {
        "executor_episodes": jnp.zeros(1, dtype=jnp.int32),
        "executor_steps": jnp.zeros(1, dtype=jnp.int32),
    }

    ############################set_parameters test#####################################
    params = {
        "trainer_steps": jnp.zeros(2, dtype=jnp.int32),
        "trainer_walltime": jnp.zeros(2, dtype=jnp.float32),
        "evaluator_steps": jnp.zeros(2, dtype=jnp.int32),
        "evaluator_episodes": jnp.zeros(2, dtype=jnp.int32),
        "executor_episodes": jnp.zeros(2, dtype=jnp.int32),
        "executor_steps": jnp.zeros(2, dtype=jnp.int32),
    }

    parameter_server.set_parameters(params)
    list_param = [
        "trainer_steps",
        "trainer_walltime",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    ]
    for param in list_param:
        assert jnp.array_equal(
            parameter_server.store.parameters[param], jnp.array([0, 0])
        )

    # set non existing param
    param1: Dict[str, str] = {"wrong_param": "test"}
    with pytest.raises(AssertionError):
        assert parameter_server.set_parameters(param1)

    ############################add_to_parameters_test#####################################
    param2: Dict[str, Any] = {
        "trainer_steps": jnp.array([1, 3]),
    }
    parameter_server.add_to_parameters(param2)

    assert jnp.array_equal(
        parameter_server.store.parameters["trainer_steps"], jnp.array([1, 3])
    )  # [0,0]+[1,3]

    ###################################step_test##########################################
    # before running step
    assert not parameter_server.store.last_checkpoint_time
    assert parameter_server.store.system_checkpointer._last_saved == 0
    # run step function
    parameter_server.store.global_config.non_blocking_sleep_seconds = 0
    parameter_server.step()

    assert parameter_server.store.last_checkpoint_time
    assert parameter_server.store.last_checkpoint_time < time.time()
    assert parameter_server.store.system_checkpointer._last_saved != 0
    assert parameter_server.store.system_checkpointer._last_saved < time.time()

    # stop the launchpad
    os.kill(pid, signal.SIGTERM)
