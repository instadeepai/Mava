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

"""Tests for trainer class for Jax-based Mava systems"""
# import functools

# import acme
# import pytest

# from mava.components.jax.building.adders import (
#     ParallelSequenceAdder,
#     ParallelSequenceAdderSignature,
# )
# from mava.components.jax.building.data_server import OnPolicyDataServer
# from mava.components.jax.building.distributor import Distributor
# from mava.components.jax.building.networks import DefaultNetworks
# from mava.components.jax.building.datasets import TrajectoryDataset
# from mava.components.jax.building.parameter_client import ExecutorParameterClient, TrainerParameterClient
# from mava.components.jax.updating.parameter_server import DefaultParameterServer
# from mava.specs import DesignSpec
# from mava.systems.jax import mappo
# from mava.systems.jax.mappo import EXECUTOR_SPEC, TRAINER_SPEC
# from mava.systems.jax.mappo.components import ExtrasLogProbSpec
# from mava.systems.jax.system import System
# from mava.testing.building import mocks
# from mava.utils.environments import debugging_utils

# #########################################################################
# # Full system integration test.
# class TestFullSystem(System):
#     def design(self) -> DesignSpec:
#         """Mock system design with zero components.

#         Returns:
#             system callback components
#         """
#         executor = EXECUTOR_SPEC.get()
#         trainer = TRAINER_SPEC.get()
#         components = DesignSpec(
#             data_server=OnPolicyDataServer,
#             data_server_adder_signature=ParallelSequenceAdderSignature,
#             extras_spec=ExtrasLogProbSpec,
#             parameter_server=DefaultParameterServer,
#             executor_parameter_client=ExecutorParameterClient,
#             **executor,
#             executor_environment_loop=mocks.MockExecutorEnvironmentLoop,
#             executor_adder=ParallelSequenceAdder,
#             networks=DefaultNetworks,
#             **trainer,
#             distributor=Distributor,
#             trainer_parameter_client=TrainerParameterClient,
#             trainer_dataset=TrajectoryDataset,
#             logger=mocks.MockLogger,
#         )
#         return components

# @pytest.fixture
# def test_full_system() -> System:
#     """Add description here."""
#     return TestFullSystem()


# def test_except_trainer(
#     test_full_system: System,
# ) -> None:
#     """Test if the parameter server instantiates processes as expected."""

#     # Environment.
#     environment_factory = functools.partial(
#         debugging_utils.make_environment,
#         env_name="simple_spread",
#         action_space="discrete",
#     )

#     # Networks.
#     network_factory = mappo.make_default_networks

#     # Build the system
#     test_full_system.build(
#         environment_factory=environment_factory,
#         network_factory=network_factory,
#         executor_parameter_update_period=20,
#         multi_process=False,
#         run_evaluator=True,
#         num_executors=1,
#         use_next_extras=False,
#     )

#     (
#         data_server,
#         parameter_server,
#         executor,
#         evaluator,
#         trainer,
#     ) = test_full_system._builder.store.system_build

#     assert isinstance(executor, acme.core.Worker)

#     # Step the executor
#     executor.run_episode()

#     # Step the trainer
#     trainer.step()