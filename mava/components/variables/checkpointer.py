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

"""Generic variable checkpointer component for systems"""
import os
import time
import abc

from mava.core import SystemVariableServer
from mava.callbacks import Callback


class Checkpointer(Callback):
    """A variable checkpointer for checkpointing variables."""

    def __init__(
        self,
        checkpoint_subpath: str,
        checkpoint_minute_interval: int,
    ) -> None:
        """[summary]

        Args:
            checkpoint (bool): [description]
            checkpoint_subpath (str): [description]
            checkpoint_minute_interval (int): [description]
        """
        self._checkpoint_subpath = checkpoint_subpath
        self._checkpoint_minute_interval = checkpoint_minute_interval
        self._last_checkpoint_time = time.time()

    def on_variables_server_init_checkpointing(
        self, server: SystemVariableServer
    ) -> None:
        # Only save variables that are not empty.
        server.save_variables = {}
        for key in server._variables.keys():
            var = server._variables[key]
            # Don't store empty tuple (e.g. empty observation_network) variables
            if not (type(var) == tuple and len(var) == 0):
                server.save_variables[key] = server._variables[key]

        # Checkpointer settings
        server.checkpoint_subpath = self._checkpoint_subpath
        server.checkpoint_subdir = os.path.join("variable_source")
        server.checkpoint_time_interval = self._checkpoint_minute_interval

    @abc.abstractmethod
    def on_variables_server_init_make_checkpointer(
        self, server: SystemVariableServer
    ) -> None:
        """[summary]"""

    def on_variables_run_server_loop_checkpoint(
        self, server: SystemVariableServer
    ) -> None:
        # Add 1 extra second just to make sure that the checkpointer
        # is ready to save.
        if (
            self._last_checkpoint_time + server.checkpoint_minute_interval * 60 + 1
            < time.time()
        ):
            server.system_checkpointer.save()
            print("Updated variables checkpoint.")
