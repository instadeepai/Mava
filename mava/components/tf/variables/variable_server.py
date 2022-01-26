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

"""Mava variable server implementation."""


import abc
from typing import Dict, Any

from acme.tf import utils as tf2_utils

from mava.core import SystemVariableServer
from mava.callbacks import Callback


class VariableServer(Callback):
    def __init__(
        self,
        variables: Dict[str, Any],
    ) -> None:
        """Initialise the variable server"""

        self.variables = variables

    def on_variables_get_server_variables(self, server: SystemVariableServer) -> None:
        """[summary]"""
        server.variables[server._var_key] = tf2_utils.to_numpy(
            self.variables[server._var_key]
        )

    def on_variables_set_server_variables_if_tuple(
        self, server: SystemVariableServer
    ) -> None:
        """[summary]"""
        server.variables[server._var_key][server._var_i].assign(
            vars[server._var_key][server._var_i]
        )

    def on_variables_set_server_variables_if_dict(
        self, server: SystemVariableServer
    ) -> None:
        """[summary]"""
        server.variables[server._var_key].assign(vars[server._var_key])

    def on_variables_add_to_server_variables(
        self, server: SystemVariableServer
    ) -> None:
        """[summary]"""
        server.variables[server._var_key].assign_add(vars[server._var_key])
