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

"""Jax systems parameter client."""

from typing import Any, Dict, List

from mava.callbacks import Callback, CallbackHookMixin
from mava.core_jax import SystemParameterClient

# import jax.numpy as jnp


class VariableClient(SystemParameterClient, CallbackHookMixin):
    def __init__(
        self,
        components: List[Callback],
    ):
        """A parameter client for pushing and pulling parameters from a server."""
        super().__init__()

        self.callbacks = components

        self.on_parameter_client_init_start()

        self.on_parameter_client_init()

        self.on_parameter_client_init_end()

    def get_async(self) -> None:
        """Asynchronously updates the get parameters with the \
            latest copy from server."""
        self.on_parameter_client_get_start()

        self.on_parameter_client_get()

        self.on_parameter_client_get_end()

    def set_async(self) -> None:
        """Asynchronously updates server with the set parameters."""
        self.on_parameter_client_set_start()

        self.on_parameter_client_set()

        self.on_parameter_client_set_end()

    def set_and_get_async(self) -> None:
        """Asynchronously updates server and gets from server."""
        self.on_parameter_client_set_and_get_start()

        self.on_parameter_client_set_and_get()

        self.on_parameter_client_set_and_get_end()

    def add_async(self, names: List[str], vars: Dict[str, Any]) -> None:
        """Asynchronously adds to server parameters."""
        self._names = names
        self._vars = vars

        self.on_parameter_client_add_start()

        self.on_parameter_client_add()

        self.on_parameter_client_add_end()

    def add_and_wait(self, names: List[str], vars: Dict[str, Any]) -> None:
        """Adds the specified parameters to the corresponding parameters in server \
        and waits for the process to complete before continuing."""
        self.on_parameter_client_add_and_wait_start()

        self.on_parameter_client_add_and_wait()

        self.on_parameter_client_add_and_wait_end()

    def get_and_wait(self) -> None:
        """Updates the get parameters with the latest copy from server \
        and waits for the process to complete before continuing."""
        self.on_parameter_client_get_and_wait_start()

        self.on_parameter_client_get_and_wait()

        self.on_parameter_client_get_and_wait_end()

    def get_all_and_wait(self) -> None:
        """Updates all the parameters with the latest copy from server \
        and waits for the process to complete before continuing."""
        self.on_parameter_client_get_all_and_wait_start()

        self.on_parameter_client_get_all_and_wait()

        self.on_parameter_client_get_all_and_wait_end()

    def set_and_wait(self) -> None:
        """Updates server with the set parameters \
        and waits for the process to complete before continuing."""
        self.on_parameter_client_set_and_wait_start()

        self.on_parameter_client_set_and_wait()

        self.on_parameter_client_set_and_wait_end()

    # def _copy(self, new_parameters: Dict[str, Any]) -> None:
    #     """Copies the new parameters to the old ones."""
    #     for key in new_parameters.keys():
    #         var_type = type(new_parameters[key])
    #         if isinstance(var_type, dict):
    #             for agent_key in new_parameters[key].keys():
    #                 for i in range(len(self._parameters[key][agent_key])):
    #                     self._parameters[key][agent_key][i] = new_parameters[key][
    #                         agent_key
    #                     ][i]
    #         elif isinstance(var_type, (jnp.int32, jnp.float32)):
    #             self._parameters[key] = new_parameters[key]
    #         elif isinstance(var_type, tuple):
    #             for i in range(len(self._parameters[key])):
    #                 self._parameters[key][i] = new_parameters[key][i]
    #         else:
    #             NotImplementedError(f"Parameter type of {var_type} not implemented.")
