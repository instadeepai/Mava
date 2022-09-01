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

import time
from typing import Any, Callable, Dict, Optional

from acme import core
from acme.jax import savers
from chex import dataclass

from mava.components.jax import Component
from mava.core_jax import SystemParameterServer

"""Checkpointer component for Mava systems."""


@dataclass
class CheckpointerConfig:
    checkpoint_minute_interval_comp: int = 5
    experiment_path: str = "~/mava/"


class SaveableWrapper(core.Saveable):
    def __init__(self, state: Dict[str, Any]):
        """Initialise system state

        Args:
            state (_type_):  system state represented by a dictionary of saved variables
        """
        self.state = state

    def save(self) -> Dict[str, Any]:
        """Save system state

        Returns:
            system state.
        """
        # TODO fix type
        return self.state

    def restore(self, state: Dict[str, Any]) -> None:
        """Restore system state

        Args:
            state (Any): system state represented by a dictionary of saved variables
        Returns:
            None.
        """
        self.state = state


class Checkpointer(Component):  # , core.Saveable):
    def __init__(
        self,
        config: CheckpointerConfig = CheckpointerConfig(),
    ):
        """Component for checkpointing system variables."""
        self.config = config
        # self._state = None

    def on_parameter_server_init(self, server: SystemParameterServer) -> None:
        """Create the system checkpointer.

        Args:
            server: SystemParameterServer.

        Returns:
            None.
        """
        # self._state = server.store.parameters
        server.store.system_checkpointer = savers.Checkpointer(
            object_to_save=SaveableWrapper(
                server.store.parameters
            ),  # must be saveable type
            directory=self.config.experiment_path,
            add_uid=False,
            time_delta_minutes=0,
        )
        server.store.last_checkpoint_time = time.time()

    def on_parameter_server_run_loop_checkpoint(
        self, server: SystemParameterServer
    ) -> None:
        """Intermittently checkpoint the server parameters.

        Args:
            server: SystemParameterServer.

        Returns:
            None.
        """
        if (
            server.store.last_checkpoint_time
            + self.config.checkpoint_minute_interval_comp * 60
            + 1
            < time.time()
        ):
            server.store.system_checkpointer._checkpoint.saveable._object_to_save = (
                SaveableWrapper(server.store.parameters)
            )
            server.store.system_checkpointer.save()
            server.store.last_checkpoint_time = time.time()

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "checkpointer"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return CheckpointerConfig

    # def save(self):
    #     return self._state

    # def restore(self, state):
    #     self._state = state
