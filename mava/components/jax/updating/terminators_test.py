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

from dataclasses import dataclass
import pytest 
from typing import List, Dict, Any
from .terminators import (
    ParameterServerTerminatorConfig, 
    ParameterServerTerminator,
)
import numpy as np 

# @pytest.fixture
def executor_steps_termination_condition() -> Dict[str, Any]:
    """Add description here."""
    return {'executor_steps': 10}

@dataclass
class MockParameterServer:
    store: str = None

@dataclass
class MockParameterStore:
    parameters: str = None

@dataclass
class MockParameterServerTerminatorConfig:
    termination_condition: str = None


# @pytest.fixture
def create_mock_parameter_server():
    """Add description here."""
    return MockParameterServer(
        store = MockParameterStore(
            parameters = {
                "trainer_steps": np.zeros(1, dtype=np.int32),
                "trainer_walltime": np.zeros(1, dtype=np.float32),
                "evaluator_steps": np.zeros(1, dtype=np.int32),
                "evaluator_episodes": np.zeros(1, dtype=np.int32),
                "executor_episodes": np.zeros(1, dtype=np.int32),
                "executor_steps": np.zeros(1, dtype=np.int32),
            }
            
        )
    )

def step_parameters(parameter_dict, key):
    parameter_dict[key] += 1

def test_executor_steps_termination_condition() -> None:

    test_terminator = ParameterServerTerminator(config=MockParameterServerTerminatorConfig(termination_condition = executor_steps_termination_condition()))

    test_parameter_server = create_mock_parameter_server()

    for _ in range(5):
        step_parameters(test_parameter_server.store.parameters, 'executor_steps')
    

    test_terminator.on_parameter_server_run_loop_termination(test_parameter_server)


# class ParameterServerTerminator(Terminator):
#     def __init__(
#         self,
#         config: ParameterServerTerminatorConfig = ParameterServerTerminatorConfig(),
#     ):
#         """_summary_

#         Args:
#             config : _description_.
#         """
#         self.config = config

#         if self.config.termination_condition is not None:
#             self.termination_key, self.termination_value = check_count_condition(
#                 self.config.termination_condition
#             )

#     def on_parameter_server_run_loop_termination(
#         self, parameter_sever: SystemParameterServer
#     ) -> None:
#         """_summary_"""
#         if (
#             self.config.termination_condition is not None
#             and parameter_sever.store.parameters[self.termination_key]
#             > self.termination_value
#         ):
#             print(
#                 f"Max {self.termination_key} of {self.termination_value}"
#                 " reached, terminating."
#             )
#             lp.stop()