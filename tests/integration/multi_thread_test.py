# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

"""Integration test of the Trainer for Jax-based Mava"""

import pytest

from mava.systems import System
from tests.systems.systems_test_data import ippo_system_multi_thread


@pytest.fixture
def test_system_mt() -> System:
    """A multi threaded built system that uses Launchpad"""
    return ippo_system_multi_thread()


def test_system_multi_thread(test_system_mt: System) -> None:
    """Test if the trainer instantiates processes as expected."""
    # Disable the nodes
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
    (data_server_node,) = test_system_mt._builder.store.program._program._groups[
        "data_server"
    ]

    trainer_node.disable_run()
    executor_node.disable_run()
    evaluator_node.disable_run()
    parameter_server_node.disable_run()
    # Data server runs in background

    # launch the system
    test_system_mt.launch()

    # Extract the instances of each node
    trainer_run = trainer_node.create_handle().dereference()
    executor_run = executor_node.create_handle().dereference()
    evaluator_run = evaluator_node.create_handle().dereference()
    parameter_server_run = parameter_server_node.create_handle().dereference()

    # Generate data with executor, run train step,
    # update parameters and finally, run an evaluation episode
    executor_run.run_environment_episode()
    trainer_run.step()
    parameter_server_run.step()
    evaluator_run.run_environment_episode()
