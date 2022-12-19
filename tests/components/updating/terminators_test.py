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

"""Terminator component unit tests"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Type

import numpy as np
import pytest

from mava.components.updating.terminators import (
    CountConditionTerminator,
    CountConditionTerminatorConfig,
    TimeTerminator,
    TimeTerminatorConfig,
)
from mava.core_jax import SystemParameterServer
from tests.components.updating.terminators_test_data import (
    count_condition_terminator_data,
    count_condition_terminator_failure_cases,
)


@dataclass
class MockParameterStore:
    """Mock for the parameter store"""

    parameters: Optional[Dict[str, Any]] = None
    stopped: bool = False


@dataclass
class MockParameterServer:
    """Mock for the parameter server"""

    store: Optional[MockParameterStore] = None


@pytest.fixture
def mock_parameter_server() -> MockParameterServer:
    """Create a mock parameter server for terminator tests"""

    mock_server = MockParameterServer(
        store=MockParameterStore(
            parameters={
                "trainer_steps": np.zeros(1, dtype=np.int32),
                "trainer_walltime": np.zeros(1, dtype=np.float32),
                "evaluator_steps": np.zeros(1, dtype=np.int32),
                "evaluator_episodes": np.zeros(1, dtype=np.int32),
                "executor_episodes": np.zeros(1, dtype=np.int32),
                "executor_steps": np.zeros(1, dtype=np.int32),
            },
            stopped=False,
        )
    )

    return mock_server


def step_parameters(parameter_dict: Dict[str, Any], key: str) -> None:
    """Utility function for stepping parameters"""

    parameter_dict[key] += 1


# Reason for # type: ignore in dataclasses -
# https://mypy.readthedocs.io/en/stable/additional_features.html#caveats-known-issues
@pytest.mark.parametrize("condition", count_condition_terminator_data())
def test_count_condition_terminator_terminated(
    condition: Dict, mock_parameter_server: SystemParameterServer
) -> None:
    """Test if count condition terminator terminates"""
    test_parameter_server = mock_parameter_server

    def _set_stopped(parameter_server: MockParameterServer) -> None:
        """Stop flag"""
        test_parameter_server.store.stopped = True

    test_terminator = CountConditionTerminator(
        config=CountConditionTerminatorConfig(  # type: ignore
            termination_condition=condition, termination_function=_set_stopped
        )
    )

    for _ in range(15):
        step_parameters(
            test_parameter_server.store.parameters, list(condition.keys())[0]
        )

    test_terminator.on_parameter_server_run_loop_termination(test_parameter_server)

    assert test_parameter_server.store.stopped is True


@pytest.mark.parametrize("condition", count_condition_terminator_data())
def test_count_condition_terminator_not_terminated(
    condition: Dict, mock_parameter_server: SystemParameterServer
) -> None:
    """Test if count condition terminator does not terminate"""
    test_parameter_server = mock_parameter_server

    def _set_stopped(parameter_server: MockParameterServer) -> None:
        """Stop flag"""
        test_parameter_server.store.stopped = True

    test_terminator = CountConditionTerminator(
        config=CountConditionTerminatorConfig(  # type: ignore
            termination_condition=condition, termination_function=_set_stopped
        )
    )

    for _ in range(5):
        step_parameters(
            test_parameter_server.store.parameters, list(condition.keys())[0]
        )

    test_terminator.on_parameter_server_run_loop_termination(test_parameter_server)

    assert test_parameter_server.store.stopped is False


@pytest.mark.parametrize(
    "fail_condition,failure", count_condition_terminator_failure_cases()
)
def test_count_condition_terminator_exceptions(
    fail_condition: Dict,
    failure: Type[Exception],
    mock_parameter_server: SystemParameterServer,
) -> None:
    """Test count condition terminator exceptions"""

    with pytest.raises(failure):
        test_parameter_server = mock_parameter_server

        def _set_stopped() -> None:
            """Stop flag"""
            test_parameter_server.store.stopped = True

        test_terminator = CountConditionTerminator(
            config=CountConditionTerminatorConfig(  # type: ignore
                termination_condition=fail_condition, termination_function=_set_stopped
            )
        )

        assert (
            test_terminator.on_parameter_server_run_loop_termination(  # type:ignore
                test_parameter_server
            )
            == failure
        )


def test_time_terminator_terminated(
    mock_parameter_server: SystemParameterServer,
) -> None:
    """Test that time terminator terminates"""

    test_parameter_server = mock_parameter_server

    def _set_stopped(parameter_server: MockParameterServer) -> None:
        """Stop flag"""
        test_parameter_server.store.stopped = True

    test_terminator = TimeTerminator(
        config=TimeTerminatorConfig(run_seconds=0.0, termination_function=_set_stopped)  # type: ignore # noqa: E501
    )

    test_terminator.on_parameter_server_init(test_parameter_server)

    test_terminator.on_parameter_server_run_loop_termination(test_parameter_server)

    assert test_parameter_server.store.stopped is True


def test_time_terminator_not_terminated(
    mock_parameter_server: SystemParameterServer,
) -> None:
    """Test that time terminator does not terminate"""

    test_parameter_server = mock_parameter_server

    def _set_stopped(parameter_server: MockParameterServer) -> None:
        """Stop flag"""
        test_parameter_server.store.stopped = True

    test_terminator = TimeTerminator(
        config=TimeTerminatorConfig(run_seconds=10, termination_function=_set_stopped)  # type: ignore # noqa: E501
    )

    test_terminator.on_parameter_server_init(test_parameter_server)

    test_terminator.on_parameter_server_run_loop_termination(test_parameter_server)

    assert test_parameter_server.store.stopped is False


@pytest.mark.parametrize("condition", count_condition_terminator_data())
def test_ignore_stop_condition(
    condition: Dict, mock_parameter_server: SystemParameterServer
) -> None:
    """Test the case we have the flag calculate_absolute_metric"""

    mock_parameter_server.calculate_absolute_metric = True  # type: ignore
    test_parameter_server = mock_parameter_server

    def _set_stopped(parameter_server: MockParameterServer) -> None:
        """Stop flag"""
        test_parameter_server.store.stopped = True

    test_terminator = CountConditionTerminator(
        config=CountConditionTerminatorConfig(  # type: ignore
            termination_condition=condition, termination_function=_set_stopped
        )
    )

    for _ in range(15):
        step_parameters(
            test_parameter_server.store.parameters, list(condition.keys())[0]
        )

    test_terminator.on_parameter_server_run_loop_termination(test_parameter_server)

    assert test_parameter_server.store.stopped is False
