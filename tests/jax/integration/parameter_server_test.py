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
import time

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

    # Initial state of the parameter_server
    assert type(parameter_server.store.system_checkpointer) == savers.Checkpointer

    for network in parameter_server.store.parameters["networks-network_agent"].values():
        assert sorted(list(network.keys())) == ["b", "w"]
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

    assert not parameter_server.store.last_checkpoint_time
    assert parameter_server.store.system_checkpointer._last_saved == 0

    # test get and set parameters
    for _ in range(3):
        executor.run_episode()
    trainer.step()

    trainer_steps = parameter_server.get_parameters("trainer_steps")
    executor_episodes = parameter_server.get_parameters("executor_episodes")
    assert list(trainer_steps) == [1]  # trainer.step one time
    assert list(executor_episodes) == [3]  # run episodes three times

    network = parameter_server.get_parameters("networks-network_agent")
    for network in network.values():
        assert sorted(list(network.keys())) == ["b", "w"]
        assert jnp.size(network["w"]) != 0  # network is updated
        assert jnp.size(network["b"]) != 0

    # run step function
    parameter_server.step()

    assert parameter_server.store.last_checkpoint_time
    assert parameter_server.store.last_checkpoint_time < time.time()
    assert parameter_server.store.system_checkpointer._last_saved != 0
    assert parameter_server.store.system_checkpointer._last_saved < time.time()


def test_parameter_server_multi_thread(test_system_mt: System) -> None:
    """Test if the paraameter server instantiates processes as expected."""
    (trainer_node,) = test_system_mt._builder.store.program._program._groups["trainer"]
    (executor_node,) = test_system_mt._builder.store.program._program._groups[
        "executor"
    ]
    (evaluator_node,) = test_system_mt._builder.store.program._program._groups[
        "evaluator"
    ]
    (parameter_server_node,) = test_system_mt._builder.store.program._program._groups[
        "parameter_server"
    ]

    trainer_node.disable_run()
    executor_node.disable_run()
    evaluator_node.disable_run()
    parameter_server_node.disable_run()

    test_system_mt.launch()
    time.sleep(10)

    parameter_server = parameter_server_node._construct_instance()
    executor = executor_node._construct_instance()
    trainer = trainer_node._construct_instance()

    # Initial state of the parameter_server
    assert type(parameter_server.store.system_checkpointer) == savers.Checkpointer

    for network in parameter_server.store.parameters["networks-network_agent"].values():
        assert sorted(list(network.keys())) == ["b", "w"]
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

    assert not parameter_server.store.last_checkpoint_time
    assert parameter_server.store.system_checkpointer._last_saved == 0

    # test get and set parameters
    for _ in range(3):
        executor.run_episode()
    trainer.step()

    trainer_steps = parameter_server.get_parameters("trainer_steps")
    executor_episodes = parameter_server.get_parameters("executor_episodes")
    assert list(trainer_steps) == [1]  # trainer.step one time
    assert list(executor_episodes) == [3]  # run episodes three times

    network = parameter_server.get_parameters("networks-network_agent")
    for network in network.values():
        assert sorted(list(network.keys())) == ["b", "w"]
        assert jnp.size(network["w"]) != 0  # network is updated
        assert jnp.size(network["b"]) != 0

    # run step function
    parameter_server.step()

    assert parameter_server.store.last_checkpoint_time
    assert parameter_server.store.last_checkpoint_time < time.time()
    assert parameter_server.store.system_checkpointer._last_saved != 0
    assert parameter_server.store.system_checkpointer._last_saved < time.time()


def test_parameter_server_multi_process(test_system_mp: System) -> None:
    """Test if the paraameter server instantiates processes as expected."""

    (trainer_node,) = test_system_mp._builder.store.program._program._groups["trainer"]
    (executor_node,) = test_system_mp._builder.store.program._program._groups[
        "executor"
    ]
    (evaluator_node,) = test_system_mp._builder.store.program._program._groups[
        "evaluator"
    ]
    (parameter_server_node,) = test_system_mp._builder.store.program._program._groups[
        "parameter_server"
    ]

    trainer_node.disable_run()
    executor_node.disable_run()
    evaluator_node.disable_run()
    parameter_server_node.disable_run()

    test_system_mp.launch()

    parameter_server = parameter_server_node._construct_instance()
    executor = executor_node._construct_instance()
    trainer = trainer_node._construct_instance()

    # Initial state of the parameter_server
    assert type(parameter_server.store.system_checkpointer) == savers.Checkpointer

    for network in parameter_server.store.parameters["networks-network_agent"].values():
        assert sorted(list(network.keys())) == ["b", "w"]
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

    assert not parameter_server.store.last_checkpoint_time
    assert parameter_server.store.system_checkpointer._last_saved == 0

    # test get and set parameters
    for _ in range(3):
        executor.run_episode()

    trainer.step()

    network = parameter_server.get_parameters("networks-network_agent")
    for network in network.values():
        assert sorted(list(network.keys())) == ["b", "w"]
        assert jnp.size(network["w"]) != 0  # network is updated
        assert jnp.size(network["b"]) != 0

    # run step function
    parameter_server.step()

    assert parameter_server.store.last_checkpoint_time
    assert parameter_server.store.last_checkpoint_time < time.time()
    assert parameter_server.store.system_checkpointer._last_saved != 0
    assert parameter_server.store.system_checkpointer._last_saved < time.time()
