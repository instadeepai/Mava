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

"""Checkpointer unit test"""

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import pytest
from acme.jax import savers as acme_savers

from mava.components.updating import Checkpointer
from mava.components.updating.checkpointer import CheckpointerConfig
from mava.core_jax import SystemParameterServer


@dataclass
class MockParameterStore:
    """Mock for the parameter store"""

    parameters: Optional[Dict[str, Any]] = None
    experiment_path: Optional[str] = None


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
            },
            experiment_path=tempfile.mkdtemp(),
        ),
    )

    return mock_server


@pytest.fixture
def checkpointer() -> Checkpointer:
    """Creates a checkpointer fixture with config

    Returns:
        ParallelSequenceAdder with ParallelSequenceAdderConfig.
    """

    checkpointer = Checkpointer(
        config=CheckpointerConfig(checkpoint_minute_interval=1 / 60),  # type: ignore
    )
    return checkpointer


def test_save_restore(
    mock_parameter_server: SystemParameterServer,
    checkpointer: Checkpointer,
) -> None:
    """Test checkpointer save and restore.

    Args:
        mock_parameter_server: Fixture SystemParameterServer.
        checkpointer: Fixture Checkpointer.

    Returns:
        None
    """
    # Create checkpointer
    checkpointer.on_parameter_server_init(server=mock_parameter_server)

    system_checkpointer = mock_parameter_server.store.system_checkpointer

    # Emulate the parameters changing e.g. trainer_steps increase and then ensure that
    # the saveable has updated
    mock_parameter_server.store.parameters["trainer_steps"] = 50
    mock_parameter_server.store.parameters["trainer_steps"] += 50

    assert (
        system_checkpointer._checkpoint.saveable._object_to_save.state
        == mock_parameter_server.store.parameters
    )
    # Save modified parameters
    time.sleep(checkpointer.config.checkpoint_minute_interval * 60 + 2)
    checkpointer.on_parameter_server_run_loop_checkpoint(server=mock_parameter_server)
    saved_trainer_steps = mock_parameter_server.store.parameters["trainer_steps"]

    # Check whether the checkpointer has saved to disk
    assert any(
        fname == "checkpoint"
        for fname in os.listdir(system_checkpointer._checkpoint_dir)
    )

    mock_parameter_server.store.parameters["trainer_steps"] += 50

    assert (
        mock_parameter_server.store.parameters["trainer_steps"] != saved_trainer_steps
    )

    # Restore the saved parameters
    system_checkpointer.restore()
    assert (
        mock_parameter_server.store.parameters["trainer_steps"] == saved_trainer_steps
    )
    assert (
        system_checkpointer._checkpoint.saveable._object_to_save.state
        == mock_parameter_server.store.parameters
    )


def test_checkpointer(
    mock_parameter_server: SystemParameterServer,
    checkpointer: Checkpointer,
) -> None:
    """Test checkpointer callbacks.

    Args:
        mock_parameter_server: Fixture SystemParameterServer.
        checkpointer: Fixture Checkpointer.

    Returns:
        None
    """
    # Check whether checkpointer parameters are set correctly
    assert checkpointer.config.checkpoint_minute_interval == 1 / 60

    # Create checkpointer
    checkpointer.on_parameter_server_init(server=mock_parameter_server)
    assert hasattr(mock_parameter_server.store, "system_checkpointer")
    assert hasattr(mock_parameter_server.store, "last_checkpoint_time")
    assert hasattr(mock_parameter_server.store, "checkpoint_minute_interval")
    assert (
        mock_parameter_server.store.checkpoint_minute_interval
        == checkpointer.config.checkpoint_minute_interval
    )

    system_checkpointer = mock_parameter_server.store.system_checkpointer
    assert type(system_checkpointer) == acme_savers.Checkpointer
    assert checkpointer.name() == "checkpointer"

    # check that checkpoint has not yet saved
    assert mock_parameter_server.store.system_checkpointer._last_saved == 0
    checkpoint_init_time = mock_parameter_server.store.last_checkpoint_time

    # Sleep until checkpoint_minute_interval elapses
    time.sleep(checkpointer.config.checkpoint_minute_interval * 60 + 2)
    checkpointer.on_parameter_server_run_loop_checkpoint(server=mock_parameter_server)

    assert mock_parameter_server.store.last_checkpoint_time > checkpoint_init_time
    assert mock_parameter_server.store.last_checkpoint_time < time.time()
    assert mock_parameter_server.store.system_checkpointer._last_saved != 0
    assert mock_parameter_server.store.system_checkpointer._last_saved < time.time()
