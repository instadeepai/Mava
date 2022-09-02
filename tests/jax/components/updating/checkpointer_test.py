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

import os
import tempfile
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pytest

from mava.components.jax.updating import Checkpointer
from mava.components.jax.updating.checkpointer import CheckpointerConfig
from mava.core_jax import SystemParameterServer


@dataclass
class MockParameterStore:
    saveable_parameters: Optional[Dict[str, Any]] = None


@dataclass
class MockParameterServer:
    store: Optional[MockParameterStore] = None


@pytest.fixture
def mock_parameter_server() -> MockParameterServer:
    """Create a mock parameter server for terminator tests"""

    mock_server = MockParameterServer(
        store=MockParameterStore(
            saveable_parameters={
                "trainer_steps": 50,
            },
        )
    )

    return mock_server


@pytest.fixture
def checkpointer() -> Checkpointer:
    """Creates a checkpointer fixture with config

    Returns:
        ParallelSequenceAdder with ParallelSequenceAdderConfig.
    """

    checkpointer = Checkpointer(
        config=CheckpointerConfig(  # type: ignore
            checkpoint_minute_interval=1 / 60, experiment_path=tempfile.mkdtemp()
        ),
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

    # Save Initial parameters and check that the file has been saved to disk
    system_checkpointer.save()
    saved_trainer_steps = mock_parameter_server.store.saveable_parameters[
        "trainer_steps"
    ]
    created_checkpoint = False
    for fname in os.listdir(
        os.path.join(checkpointer.config.experiment_path, "checkpoints/default")
    ):
        if fname == "checkpoint":
            created_checkpoint = True
            break
    assert created_checkpoint

    # Change the parameters and check that the checkpointer has the same value
    mock_parameter_server.store.saveable_parameters["trainer_steps"] += 50

    assert (
        system_checkpointer._checkpoint.saveable._object_to_save.state
        == mock_parameter_server.store.saveable_parameters
    )

    # Restore the saved parameters and check that the checkpointer values are correct
    system_checkpointer.restore()
    assert (
        system_checkpointer._checkpoint.saveable._object_to_save.state
        != mock_parameter_server.store.saveable_parameters
    )
    assert (
        system_checkpointer._checkpoint.saveable._object_to_save.state["trainer_steps"]
        == saved_trainer_steps
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
    assert "/tmp" in checkpointer.config.experiment_path

    # Create checkpointer
    checkpointer.on_parameter_server_init(server=mock_parameter_server)
    assert mock_parameter_server.store.system_checkpointer
    assert mock_parameter_server.store.last_checkpoint_time
    system_checkpointer = mock_parameter_server.store.system_checkpointer

    # Emulate parameters changing, as per system i.e. trainer_steps increase
    mock_parameter_server.store.saveable_parameters = {"trainer_steps": 100}

    # Allow for checkpoint_minute_interval to elapse
    time.sleep(checkpointer.config.checkpoint_minute_interval * 60 + 1)
    checkpointer.on_parameter_server_run_loop_checkpoint(server=mock_parameter_server)

    # Check whether the checkpointer has saved
    assert (
        mock_parameter_server.store.saveable_parameters["trainer_steps"]
        == system_checkpointer._checkpoint.saveable._object_to_save.state[
            "trainer_steps"
        ]
    )

    assert checkpointer.name() == "checkpointer"
