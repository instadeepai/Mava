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

import acme
import pytest

from mava.components.jax.building.parameter_client import DefaultParameterClient
from mava.components.jax.updating.parameter_server import DefaultParameterServer
from mava.specs import DesignSpec
from mava.systems.jax import mappo
from mava.systems.jax.mappo import EXECUTOR_SPEC
from mava.systems.jax.system import System
from mava.testing.building import mocks
from mava.utils.environments import debugging_utils


# Test executor in isolation.
class TestSystem(System):
    def design(self) -> DesignSpec:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        executor = EXECUTOR_SPEC.get()
        components = DesignSpec(
            data_server=mocks.MockDataServer,
            data_server_adder=mocks.MockAdderSignature,
            parameter_server=mocks.MockParameterServer,
            parameter_client=mocks.MockParameterClient,
            logger=mocks.MockLogger,
            **executor,
            executor_environment_loop=mocks.MockExecutorEnvironmentLoop,
            executor_adder=mocks.MockAdder,
            networks=mocks.MockNetworks,
            trainer=mocks.MockTrainer,
            trainer_dataset=mocks.MockTrainerDataset,
            distributor=mocks.MockDistributor,
        )
        return components


@pytest.fixture
def test_system() -> System:
    """Dummy system with zero components."""
    return TestSystem()


def test_executor(
    test_system: System,
) -> None:
    """Test if the parameter server instantiates processes as expected."""
    system = TestSystem()

    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    network_factory = mappo.make_default_networks

    # Build the system
    system.build(
        environment_factory=environment_factory, network_factory=network_factory
    )

    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = system._builder.config.system_build

    assert isinstance(executor, acme.core.Worker)

    # Run an episode
    executor.run_episode()


# Intergration test between the executor, variable_client and variable_server.
# class TestSystem(System):
#     def design(self) -> DesignSpec:
#         """Mock system design with zero components.

#         Returns:
#             system callback components
#         """
#         executor = EXECUTOR_SPEC.get()
#         components = DesignSpec(
#             data_server=mocks.MockDataServer,
#             data_server_adder=mocks.MockAdderSignature,
#             parameter_server=DefaultParameterServer,
#             parameter_client=DefaultParameterClient,
#             logger=mocks.MockLogger,
#             **executor,
#             executor_environment_loop=mocks.MockExecutorEnvironmentLoop,
#             executor_adder=mocks.MockAdder,
#             networks=mocks.MockNetworks,
#             trainer=mocks.MockTrainer,
#             trainer_dataset=mocks.MockTrainerDataset,
#             distributor=mocks.MockDistributor,
#         )
#         return components


# @pytest.fixture
# def test_system() -> System:
#     """Dummy system with zero components."""
#     return TestSystem()


# def test_executor(
#     test_system: System,
# ) -> None:
#     """Test if the parameter server instantiates processes as expected."""
#     system = TestSystem()

#     # Environment.
#     environment_factory = functools.partial(
#         debugging_utils.make_environment,
#         env_name="simple_spread",
#         action_space="discrete",
#     )

#     # Networks.
#     network_factory = mappo.make_default_networks

#     # Build the system
#     system.build(
#         environment_factory=environment_factory, network_factory=network_factory
#     )

#     (
#         data_server,
#         parameter_server,
#         executor,
#         evaluator,
#         trainer,
#     ) = system._builder.config.system_build

#     assert isinstance(executor, acme.core.Worker)

#     # Run an episode
#     executor.run_episode()
