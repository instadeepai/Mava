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

from mava.core import SystemVariableServer
from mava.components.variables import VariableCheckpointer as BaseCheckpointer
from mava.systems.tf import savers as tf2_savers


class VariableCheckpointer(BaseCheckpointer):
    """A variable checkpointer for checkpointing variables."""

    def on_variables_server_init_make_checkpointer(
        self, server: SystemVariableServer
    ) -> None:
        """[summary]"""
        server.system_checkpointer = tf2_savers.Checkpointer(
            time_delta_minutes=server.checkpoint_minute_interval,
            directory=server.checkpoint_subpath,
            objects_to_save=server.save_variables,
            subdirectory=server.checkpoint_subdir,
        )
