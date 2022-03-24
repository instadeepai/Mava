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

from types import SimpleNamespace

import pytest

from mava.systems.jax.system import System
from mava.testing.building import mocks


class TestSystem(System):
    def design(self) -> SimpleNamespace:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        components = SimpleNamespace(
            data_server=mocks.MockDataServer,
            data_server_adder=mocks.MockAdderSignature,
            parameter_server=mocks.MockParameterServer,
            parameter_client=mocks.MockParameterClient,
            logger=mocks.MockLogger,
            executor=mocks.MockExecutor,
            executor_adder=mocks.MockAdder,
            executor_environment_loop=mocks.MockExecutorEnvironmentLoop,
            trainer=mocks.MockTrainer,
            trainer_dataset=mocks.MockTrainerDataset,
            distributor=mocks.MockDistributor,
        )
        return components


@pytest.fixture
def test_system() -> System:
    """Dummy system with zero components."""
    return TestSystem()


def test_builder(
    test_system: System,
) -> None:
    """Test if system builder instantiates processes as expected."""
    test_system.configure()
    test_system.launch(num_executors=1, nodes_on_gpu=["process"])
    (
        data_server,
        parameter_server,
        executor,
        evaluator,
        trainer,
    ) = test_system._builder.attr.system_build
    assert data_server == 2
    assert parameter_server == 2

    exec_data_client, env, exec = executor
    exec_logger, exec_logger_param = env
    assert exec_data_client == data_server
    assert exec_logger_param == "param"
    assert exec_logger == 1

    exec_id, exec_adder, exec_param_client, exec_param = exec
    assert exec_id == "executor"
    assert exec_param_client == (2, 1)
    assert exec_adder == 2.7
    assert exec_param == 1

    eval_env, eval_exec = evaluator
    eval_id, eval_param_client, eval_exec_param = eval_exec
    assert eval_env == env
    assert eval_id == "evaluator"
    assert eval_param_client == exec_param_client
    assert eval_exec_param == exec_param

    train_id, train_logger, train_dataset, train_param_client = trainer
    assert train_id == "trainer"
    assert train_logger == 1
    assert train_dataset == 5
    assert train_param_client == (2, "param")
