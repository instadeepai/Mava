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

"""Generic variable client component for systems"""

from acme.tf import utils as tf2_utils

from mava.components.variables import VariableClient as BaseClient
from mava.callbacks import Callback
from mava.core import SystemVariableClient


class VariableClient(BaseClient):
    """A variable client for updating variables from a remote source."""

    def _adjust_and_request(self, client: SystemVariableClient) -> None:
        client._server.set_variables(
            client._set_keys,
            tf2_utils.to_numpy(
                {key: client._variables[key] for key in client._set_keys}
            ),
        )
        client._copy(client._server.get_variables(client._get_keys))

    def on_variables_client_adjust_and_request(
        self, client: SystemVariableClient
    ) -> None:
        """[summary]"""
        client._adjust = lambda: client._server.set_variables(
            client._set_keys,
            tf2_utils.to_numpy(
                {key: client._variables[key] for key in client._set_keys}
            ),
        )
        client._add = lambda names, vars: client._server.add_to_variables(names, vars)

    def on_variables_client_copy_if_dict(self, client: SystemVariableClient) -> None:
        """[summary]"""
        for agent_key in client._new_variables[client._key].keys():
            for i in range(len(self._variables[client._key][agent_key])):
                client._variables[client._key][agent_key][i].assign(
                    client._new_variables[client._key][agent_key][i]
                )

    def on_variables_client_copy_if_int_float(
        self, client: SystemVariableClient
    ) -> None:
        """[summary]"""
        # TODO (dries): Is this count value getting tracked?
        client._variables[client._key].assign(client._new_variables[client._key])

    def on_variables_client_copy_if_tuple(self, client: SystemVariableClient) -> None:
        """[summary]"""
        for i in range(len(self._variables[client._key])):
            client._variables[client._key][i].assign(
                client._new_variables[client._key][i]
            )
