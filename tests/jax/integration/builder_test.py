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

"""Tests for config class for Jax-based Mava systems"""

from typing import Dict, Tuple

import acme
import pytest

from mava.components.building.adders import ParallelTransitionAdderSignature
from mava.components.building.environments import EnvironmentSpec
from mava.components.building.system_init import FixedNetworkSystemInit
from mava.specs import DesignSpec
from mava.systems import ParameterServer, Trainer
from mava.systems.system import System
from tests.jax import mocks


class TestSystem(System):
    def design(self) -> Tuple[DesignSpec, Dict]:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        components = DesignSpec(
            system_init=FixedNetworkSystemInit,
            environment_spec=EnvironmentSpec,
            data_server=mocks.MockOnPolicyDataServer,
            data_server_adder_signature=ParallelTransitionAdderSignature,
            parameter_server=mocks.MockParameterServer,
            executor_parameter_client=mocks.MockExecutorParameterClient,
            trainer_parameter_client=mocks.MockTrainerParameterClient,
            logger=mocks.MockLogger,
            executor=mocks.MockExecutor,
            executor_adder=mocks.MockAdder,
            executor_environment_loop=mocks.MockExecutorEnvironmentLoop,
            trainer=mocks.MockTrainer,
            trainer_dataset=mocks.MockTrainerDataset,
            distributor=mocks.MockDistributor,
        )
        return components, {}


@pytest.fixture
def test_system() -> System:
    """Dummy system with zero components."""
    return TestSystem()


# TODO Rewrite test
def test_builder(
    test_system: System,
) -> None:
    """Test if system builder instantiates processes as expected."""
    test_system.build(
        environment_factory=mocks.make_fake_environment_factory(),
        trainer_parameter_update_period=1,
    )
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system._builder.store.system_build

    assert isinstance(parameter_server, ParameterServer)

    assert isinstance(executor, acme.core.Worker)
    # exec_data_client, env, exec = executor
    # exec_logger, exec_logger_param = env
    # assert exec_data_client == data_server
    # assert exec_logger_param == "param"
    # assert exec_logger == 1

    # exec_id, exec_adder, exec_param_client, exec_param = exec
    # assert exec_id == "executor"
    # assert type(exec_param_client[0]) == ParameterServer
    # assert exec_param_client[1] == 1
    # assert exec_adder == 2.7
    # assert exec_param == 1

    assert isinstance(evaluator, acme.core.Worker)
    # eval_env, eval_exec = evaluator
    # eval_id, eval_param_client, eval_exec_param = eval_exec
    # assert eval_env == env
    # assert eval_id == "evaluator"
    # assert eval_param_client == exec_param_client
    # assert eval_exec_param == exec_param

    assert isinstance(trainer, Trainer)
    # assert train_id == "trainer_0"
    # assert train_logger == 1
    # assert train_dataset == 5
    # assert train_param_client == (2, "param")
