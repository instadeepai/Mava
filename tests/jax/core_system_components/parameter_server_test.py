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

import functools
import hashlib
from types import SimpleNamespace
from typing import Dict, List, Tuple

import numpy as np
import pytest

from mava.callbacks import Callback
from mava.components.jax import building
from mava.components.jax.building.adders import ParallelTransitionAdderSignature
from mava.components.jax.updating.parameter_server import DefaultParameterServer
from mava.specs import DesignSpec
from mava.systems.jax import ParameterServer, mappo
from mava.systems.jax.system import System
from mava.utils.environments import debugging_utils
from tests.jax import mocks


class TestSystem(System):
    def design(self) -> Tuple[DesignSpec, Dict]:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        components = DesignSpec(
            environment_spec=building.EnvironmentSpec,
            system_init=building.SystemInit,
            data_server=mocks.MockOnPolicyDataServer,
            data_server_adder_signature=ParallelTransitionAdderSignature,
            parameter_server=DefaultParameterServer,
            executor_parameter_client=mocks.MockExecutorParameterClient,
            trainer_parameter_client=mocks.MockTrainerParameterClient,
            logger=mocks.MockLogger,
            executor=mocks.MockExecutor,
            executor_adder=mocks.MockAdder,
            executor_environment_loop=mocks.MockExecutorEnvironmentLoop,
            networks=mocks.MockNetworks,
            trainer=mocks.MockTrainer,
            trainer_dataset=mocks.MockTrainerDataset,
            distributor=mocks.MockDistributor,
        )
        return components, {}


class TestParameterServer(ParameterServer):
    initial_token_value = "initial_token_value"

    def __init__(
        self,
        config: SimpleNamespace,
        components: List[Callback],
    ) -> None:
        """Initialise the parameter server."""
        self.token = self.initial_token_value

        super().__init__(config, components)

    def reset_token(self) -> None:
        """Reset token to initial value"""
        self.token = self.initial_token_value

    # init hooks
    def on_parameter_server_init_start(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_init_start")

    def on_parameter_server_init(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_init")

    def on_parameter_server_init_checkpointer(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_init_checkpointer")

    def on_parameter_server_init_end(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_init_end")

    # get_parameters hooks
    def on_parameter_server_get_parameters_start(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_get_parameters_start")

    def on_parameter_server_get_parameters(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_get_parameters")

    def on_parameter_server_get_parameters_end(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_get_parameters_end")

    # set_parameters hooks
    def on_parameter_server_set_parameters_start(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_set_parameters_start")

    def on_parameter_server_set_parameters(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_set_parameters")

    def on_parameter_server_set_parameters_end(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_set_parameters_end")

    # add_to_parameters hooks
    def on_parameter_server_add_to_parameters_start(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(
            self.token, "on_parameter_server_add_to_parameters_start"
        )

    def on_parameter_server_add_to_parameters(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_add_to_parameters")

    def on_parameter_server_add_to_parameters_end(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_add_to_parameters_end")

    # step hooks
    def on_parameter_server_run_loop_start(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_run_loop_start")

    def on_parameter_server_run_loop_checkpoint(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_run_loop_checkpoint")

    def on_parameter_server_run_loop(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_run_loop")

    def on_parameter_server_run_loop_termination(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_run_loop_termination")

    def on_parameter_server_run_loop_end(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_run_loop_end")

    # run hooks
    def on_parameter_server_run_start(self) -> None:
        """Override hook to update token using the method name"""
        self.token = hash_token(self.token, "on_parameter_server_run_start")


@pytest.fixture
def test_system() -> System:
    """Dummy system with zero components."""
    return TestSystem()


@pytest.fixture
def test_parameter_server() -> ParameterServer:
    """Dummy parameter server with no components"""
    return TestParameterServer(SimpleNamespace(config_key="expected_value"), [])


def hash_token(token: str, hash_by: str) -> str:
    """Use 'hash_by' to hash the given string token"""
    return hashlib.md5((token + hash_by).encode()).hexdigest()


def get_final_token_value(method_names: List[str]) -> str:
    """Get the final expected value of a token after it is hashed by the method names"""
    token = TestParameterServer.initial_token_value
    for method_name in method_names:
        token = hash_token(token, method_name)
    return token


def test_parameter_server_process_instantiate(
    test_system: System,
) -> None:
    """Test if the parameter server instantiates processes as expected."""
    # Environment.
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    # Networks.
    network_factory = mappo.make_default_networks

    test_system.build(
        environment_factory=environment_factory,
        network_factory=network_factory,
        non_blocking_sleep_seconds=0,
    )
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system._builder.store.system_build
    assert type(parameter_server) == ParameterServer

    step_var = parameter_server.get_parameters("trainer_steps")
    assert type(step_var) == np.ndarray
    assert step_var[0] == 0

    parameter_server.set_parameters({"trainer_steps": np.ones(1, dtype=np.int32)})
    assert parameter_server.get_parameters("trainer_steps")[0] == 1

    parameter_server.add_to_parameters({"trainer_steps": np.ones(1, dtype=np.int32)})
    assert parameter_server.get_parameters("trainer_steps")[0] == 2

    # Step the parameter sever
    parameter_server.step()


def test_init_hook_order(test_parameter_server: TestParameterServer) -> None:
    """Test if init hooks are called in the correct order"""
    assert test_parameter_server.token == get_final_token_value(
        [
            "on_parameter_server_init_start",
            "on_parameter_server_init",
            "on_parameter_server_init_checkpointer",
            "on_parameter_server_init_end",
        ]
    )
