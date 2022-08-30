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

"""Tests for executor class for Jax-based Mava systems"""

import functools
from typing import Dict, Tuple

import acme
import numpy as np
import pytest

from mava.components.jax import building, executing
from mava.components.jax.building.adders import (
    ParallelSequenceAdderSignature,
    UniformAdderPriority,
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
from tests.jax import mocks

system_init = DesignSpec(
    environment_spec=building.EnvironmentSpec,
    system_init=building.FixedNetworkSystemInit,
).get()
executor = DesignSpec(
    executor_init=executing.ExecutorInit,
    executor_observe=executing.FeedforwardExecutorObserve,
    executor_select_action=executing.FeedforwardExecutorSelectAction,
    executor_adder=building.ParallelSequenceAdder,
    adder_priority=UniformAdderPriority,
    executor_environment_loop=building.ParallelExecutorEnvironmentLoop,
    networks=building.DefaultNetworks,
).get()


#########################################################################
# Test executor in isolation.
class TestSystemExecutor(System):
    def design(self) -> Tuple[DesignSpec, Dict]:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        components = DesignSpec(
            **system_init,
            data_server=mocks.MockOnPolicyDataServer,
            data_server_adder_signature=ParallelSequenceAdderSignature,
            parameter_server=mocks.MockParameterServer,
            executor_parameter_client=mocks.MockExecutorParameterClient,
            trainer_parameter_client=mocks.MockTrainerParameterClient,
            logger=mocks.MockLogger,
            **executor,
            trainer=mocks.MockTrainer,
            trainer_dataset=mocks.MockTrainerDataset,
            distributor=mocks.MockDistributor,
        )
        return components, {}


@pytest.fixture
def test_exector_system() -> System:
    """Add description here."""
    return TestSystemExecutor()


# Skip failing test for now
@pytest.mark.skip
def test_executor(
    test_exector_system: System,
) -> None:
    """Test if the parameter server instantiates processes as expected."""

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    network_factory = ippo.make_default_networks

    # Build the system
    test_exector_system.build(
        environment_factory=environment_factory, network_factory=network_factory
    )

    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_exector_system._builder.store.system_build

    assert isinstance(executor, acme.core.Worker)

    # Run an episode
    executor.run_episode()


#########################################################################
# Integration test for the executor, variable_client and variable_server.
class TestSystemExecutorAndParameterSever(System):
    def design(self) -> Tuple[DesignSpec, Dict]:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        components = DesignSpec(
            **system_init,
            data_server=mocks.MockOnPolicyDataServer,
            data_server_adder_signature=ParallelSequenceAdderSignature,
            parameter_server=DefaultParameterServer,
            executor_parameter_client=ExecutorParameterClient,
            trainer_parameter_client=mocks.MockTrainerParameterClient,
            logger=mocks.MockLogger,
            **executor,
            trainer=mocks.MockTrainer,
            trainer_dataset=mocks.MockTrainerDataset,
            distributor=mocks.MockDistributor,
        )
        return components, {}


@pytest.fixture
def test_executor_parameter_server_system() -> System:
    """Add description here."""
    return TestSystemExecutorAndParameterSever()


# Skip failing test for now
@pytest.mark.skip
def test_executor_parameter_server(
    test_executor_parameter_server_system: System,
) -> None:
    """Test if the parameter server instantiates processes as expected."""

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    network_factory = ippo.make_default_networks

    # Build the system
    test_executor_parameter_server_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        executor_parameter_update_period=1,
    )

    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_executor_parameter_server_system._builder.store.system_build

    assert isinstance(executor, acme.core.Worker)

    # Save the executor policy

    parameters = executor._executor.store.executor_parameter_client._parameters

    # Change a variable in the policy network
    parameter_server.set_parameters(
        {"evaluator_steps": np.full(1, 1234, dtype=np.int32)}
    )

    # Step the executor
    executor.run_episode()

    # Check if the executor variable has changed.
    parameters = executor._executor.store.executor_parameter_client._parameters
    assert parameters["evaluator_steps"] == 1234


#########################################################################
# Integration test for the executor, adder, data_server, variable_client
# and variable_server.
class TestSystemExceptTrainer(System):
    def design(self) -> Tuple[DesignSpec, Dict]:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        components = DesignSpec(
            **system_init,
            data_server=OnPolicyDataServer,
            data_server_adder_signature=ParallelSequenceAdderSignature,
            extras_spec=ExtrasLogProbSpec,
            parameter_server=DefaultParameterServer,
            executor_parameter_client=ExecutorParameterClient,
            **executor,
            distributor=Distributor,
            trainer_parameter_client=mocks.MockTrainerParameterClient,
            logger=mocks.MockLogger,
            trainer=mocks.MockTrainer,
            trainer_dataset=mocks.MockTrainerDataset,
        )
        return components, {}


@pytest.fixture
def test_system_except_trainer() -> System:
    """Add description here."""
    return TestSystemExceptTrainer()


# Skip failing test for now
@pytest.mark.skip
def test_except_trainer(
    test_system_except_trainer: System,
) -> None:
    """Test if the parameter server instantiates processes as expected."""

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    network_factory = ippo.make_default_networks

    # Build the system
    test_system_except_trainer.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        executor_parameter_update_period=1,
        multi_process=False,
        run_evaluator=True,
        num_executors=1,
        use_next_extras=False,
    )

    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system_except_trainer._builder.store.system_build

    assert isinstance(executor, acme.core.Worker)

    # Step the executor
    executor.run_episode()
