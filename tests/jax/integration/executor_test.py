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

"""Integration test of The Executor for Jax-based Mava systems"""

import functools
import time
from datetime import datetime
from typing import Any

import launchpad as lp
import optax
import pytest
from launchpad.launch.test_multi_threading import (
    address_builder as test_address_builder,
)

from mava.components.jax.building.data_server import OnPolicyDataServer
from mava.components.jax.building.distributor import Distributor
from mava.components.jax.building.parameter_client import ExecutorParameterClient
from mava.components.jax.updating.parameter_server import DefaultParameterServer
from mava.specs import DesignSpec
from mava.systems.jax import ippo
from mava.systems.jax.ippo.components import ExtrasLogProbSpec
from mava.systems.jax.system import System

from mava.utils.environments import debugging_utils
from mava.utils.loggers import logger_utils


@pytest.fixture
def test_system() -> System
    """Mappo system"""
    return mappo.MAPPOSystem()


def test_executor_single_process_with_adder(test_system: System) -> None:
    """Test if the executor instantiates processes as expected."""

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return mappo.make_default_networks(  # type: ignore
            policy_layer_sizes=(32, 32),
            critic_layer_sizes=(64, 64),
            *args,
            **kwargs,
        )


    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = "~/mava"
    mava_id = str(datetime.now())
    checkpoint_subpath = f"{base_dir}/{mava_id}"
    # Log every [log_every] seconds.
    log_every = 1
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=mava_id,
        time_delta=log_every,
    )
    # Optimizer.
    optimizer = optax.chain(
        optax.clip_by_global_norm(40.0),
        optax.adam(1e-4),
    )

    # Build the test_system
    test_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=10,
        multi_process=False,
        run_evaluator=True,
        num_executors=1,
        use_next_extras=False,
        max_queue_size=5000,
        sample_batch_size=2,
        max_in_flight_samples_per_worker=4,
        num_workers_per_iterator=-1,
        rate_limiter_timeout_ms=-1,
        nodes_on_gpu=[],
        lp_launch_type=lp.LaunchType.TEST_MULTI_THREADING,
        sequence_length=4,
        period=4,
    )

    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system._builder.store.system_build

    # _writer.append needs to be called once to get _writer.history
    # _writer.append called in observe_first and observe
    with pytest.raises(RuntimeError):
        assert executor._executor.store.adder._writer.history

    # Run an episode
    executor.run_episode()

    # Observe first and observe
    assert executor._executor.store.adder._writer.history
    assert list(executor._executor.store.adder._writer.history.keys()) == [
        "observations",
        "start_of_episode",
        "actions",
        "rewards",
        "discounts",
        "extras",
    ]
    assert list(
        executor._executor.store.adder._writer.history["observations"].keys()
    ) == ["agent_0", "agent_1", "agent_2"]
    assert (
        type(executor._executor.store.adder._writer.history["observations"]["agent_0"])
        == OLT
    )

    assert len(executor._executor.store.adder._writer._column_history) != 0

    # Select actions and select action
    assert list(executor._executor.store.actions_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    assert list(executor._executor.store.policies_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    num_possible_actions = environment_factory().action_spec()["agent_0"].num_values
    assert (
        lambda: x in range(0, num_possible_actions)
        for x in list(executor._executor.store.actions_info.values())
    )
    assert (
        lambda: key == "log_prob"
        for key in executor._executor.store.policies_info.values()
    )


def test_executor_single_process_without_adder(test_system: System) -> None:
    """Test if the executor instantiates processes as expected."""
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return mappo.make_default_networks(  # type: ignore
            policy_layer_sizes=(32, 32),
            critic_layer_sizes=(64, 64),
            *args,
            **kwargs,
        )


    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = "~/mava"
    mava_id = str(datetime.now())
    checkpoint_subpath = f"{base_dir}/{mava_id}"
    # Log every [log_every] seconds.
    log_every = 1
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=mava_id,
        time_delta=log_every,
    )
    # Optimizer.
    optimizer = optax.chain(
        optax.clip_by_global_norm(40.0),
        optax.adam(1e-4),
    )

    # Build the test_system
    test_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=10,
        multi_process=False,
        run_evaluator=True,
        num_executors=1,
        use_next_extras=False,
        max_queue_size=5000,
        sample_batch_size=2,
        max_in_flight_samples_per_worker=4,
        num_workers_per_iterator=-1,
        rate_limiter_timeout_ms=-1,
        nodes_on_gpu=[],
        lp_launch_type=lp.LaunchType.TEST_MULTI_THREADING,
        sequence_length=4,
        period=4,
    )

    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system._builder.store.system_build
    # Remove adder
    executor._executor.store.adder = None
    # Run an episode
    executor.run_episode()

    # Observe first (without adder)
    assert executor._executor.store.adder is None

    # Select actions and select action
    assert list(executor._executor.store.actions_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    assert list(executor._executor.store.policies_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    num_possible_actions = environment_factory().action_spec()["agent_0"].num_values
    assert (
        lambda: x in range(0, num_possible_actions)
        for x in list(executor._executor.store.actions_info.values())
    )

    assert (
        lambda: key == "log_prob"
        for key in executor._executor.store.policies_info.values()
    )

    # Observe (without adder)
    assert not hasattr(executor._executor.store.adder, "add")


def test_executor_multi_process_with_adder(test_system: System) -> None:
    """Test if the executor instantiates processes as expected."""
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return mappo.make_default_networks(  # type: ignore
            policy_layer_sizes=(32, 32),
            critic_layer_sizes=(64, 64),
            *args,
            **kwargs,
        )

    # Checkpointer appends "Checkpoints" to checkpoint_dir.
    base_dir = "~/mava"
    mava_id = str(datetime.now())
    checkpoint_subpath = f"{base_dir}/{mava_id}"

    # Log every [log_every] seconds.
    log_every = 10
    logger_factory = functools.partial(
        logger_utils.make_logger,
        directory=base_dir,
        to_terminal=True,
        to_tensorboard=True,
        time_stamp=mava_id,
        time_delta=log_every,
    )

    # Optimizer.
    optimizer = optax.chain(
        optax.clip_by_global_norm(40.0),
        optax.adam(1e-4),
    )


    # Build the system
    test_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        logger_factory=logger_factory,
        experiment_path=checkpoint_subpath,
        optimizer=optimizer,
        executor_parameter_update_period=1,
        multi_process=True,
        run_evaluator=True,
        num_executors=1,
        max_queue_size=500,
        use_next_extras=False,
        sample_batch_size=5,
        nodes_on_gpu=[],
        lp_launch_type=lp.LaunchType.TEST_MULTI_THREADING,
    )
    (executor_node,) = test_system._builder.store.program._program._groups["executor"]
    test_address_builder.bind_addresses([executor_node])
    test_system.launch()
    time.sleep(10)
    executor = executor_node._construct_instance()

    # Observe first and observe
    assert executor._executor.store.adder._writer.history
    assert list(executor._executor.store.adder._writer.history.keys()) == [
        "observations",
        "start_of_episode",
        "actions",
        "rewards",
        "discounts",
        "extras",
    ]
    assert list(
        executor._executor.store.adder._writer.history["observations"].keys()
    ) == ["agent_0", "agent_1", "agent_2"]
    assert (
        type(executor._executor.store.adder._writer.history["observations"]["agent_0"])
        == OLT
    )

    assert len(executor._executor.store.adder._writer._column_history) != 0

    # Select actions and select action
    i = 0
    while (
        sorted(list(executor._executor.store.actions_info.keys()))
        != ["agent_0", "agent_1", "agent_2"]
        and i < 100
    ):
        time.sleep(2)
        i += 1

    assert list(executor._executor.store.actions_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    assert list(executor._executor.store.policies_info.keys()) == [
        "agent_0",
        "agent_1",
        "agent_2",
    ]
    num_possible_actions = environment_factory().action_spec()["agent_0"].num_values
    assert (
        lambda: x in range(0, num_possible_actions)
        for x in list(executor._executor.store.actions_info.values())
    )
    assert (
        lambda: key == "log_prob"
        for key in executor._executor.store.policies_info.values()
    )
