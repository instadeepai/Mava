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


from typing import Dict, Sequence, Union, List, Any

import numpy as np

from mava.core import SystemVariableServer
from mava.core import SystemVariableClient
from mava.callbacks import Callback
from mava.callbacks import CallbackHookMixin
from mava.utils.training_utils import non_blocking_sleep


class VariableServer(SystemVariableServer, CallbackHookMixin):
    def __init__(
        self,
        components: List[Callback],
    ) -> None:
        """Initialise the variable source
        Args:
            variables (Dict[str, Any]): a dictionary with
            variables which should be stored in it.
            checkpoint (bool): Indicates whether checkpointing should be performed.
            checkpoint_subpath (str): checkpoint path
        Returns:
            None
        """
        self.callbacks = components

        self.on_variables_server_init_start()

        self.on_variables_server_init()

        self.on_variables_server_init_checkpointing()

        self.on_variables_server_init_make_checkpointer()

        self.on_variables_server_init_end()

    def get_variables(
        self, names: Union[str, Sequence[str]]
    ) -> Dict[str, Dict[str, np.ndarray]]:
        """Get variables from the variable source.
        Args:
            names (Union[str, Sequence[str]]): Names of the variables to get.
        Returns:
            variables(Dict[str, Dict[str, np.ndarray]]): The variables that
            were requested.
        """
        self._names = names

        self.on_variables_get_server_variables_start()

        if type(names) == str:
            variables = self.variables[names]
        else:
            variables: Dict[str, Dict[str, np.ndarray]] = {}
            for var_key in names:
                self._var_key = var_key
                # TODO (dries): Do we really have to convert the variables to
                # numpy each time. Can we not keep the variables in numpy form
                # without the checkpointer complaining?
                self.on_variables_get_server_variables()

        self.on_variables_get_server_variables_end()

        return variables

    def set_variables(self, names: Sequence[str], vars: Dict[str, np.ndarray]) -> None:
        """Set variables in the variable source.
        Args:
            names (Union[str, Sequence[str]]): Names of the variables to set.
            vars(Dict[str, np.ndarray]): The values to set the variables to.
        Returns:
            None
        """
        self._names = names
        self._vars = vars

        self.on_variables_set_server_variables_start()

        if type(names) == str:
            vars = {names: vars}  # type: ignore
            names = [names]  # type: ignore

        for var_key in names:
            self._var_key = var_key
            assert var_key in self.variables
            if type(self.variables[var_key]) == tuple:
                # Loop through tuple
                for var_i in range(len(self.variables[var_key])):
                    self._var_i = var_i
                    self.on_variables_set_server_variables_if_tuple()
            else:
                self.on_variables_set_server_variables_if_dict()

        self.on_variables_set_server_variables_end()

    def add_to_variables(
        self, names: Sequence[str], vars: Dict[str, np.ndarray]
    ) -> None:
        """Add to the variables in the variable source.
        Args:
            names (Union[str, Sequence[str]]): Names of the variables to add to.
            vars(Dict[str, np.ndarray]): The values to add to the variables to.
        Returns:
            None
        """
        self._names = names
        self._vars = vars

        self.on_variables_add_to_server_variables_start()

        if type(names) == str:
            vars = {names: vars}  # type: ignore
            names = [names]  # type: ignore

        for var_key in names:
            assert var_key in self.variables
            self._var_key = var_key
            # Note: Can also use self.variables[var_key] = /
            # self.variables[var_key] + vars[var_key]
            self.on_variables_add_to_server_variables()

        self.on_variables_add_to_server_variables_end()

    def run(self) -> None:
        """Run the variable source. This function allows for
        checkpointing and other centralised computations to
        be performed by the variable source.
                Args:
                    None
                Returns:
                    None
        """

        self.on_variables_run_server_start()

        # Checkpoints every 5 minutes
        while True:
            # Wait 10 seconds before checking again
            non_blocking_sleep(10)

            # Add 1 extra second just to make sure that the checkpointer
            # is ready to save.
            self.on_variables_run_server_loop_start()

            self.on_variables_run_server_loop_checkpoint()

            self.on_variables_run_server_loop()

            self.on_variables_run_server_loop_termination()

            self.on_variables_run_server_loop_end()


class VariableClient(SystemVariableClient, CallbackHookMixin):
    """A variable client for updating variables from a remote source."""

    def __init__(
        self,
        components: List[Callback],
    ):

        self.callbacks = components

        self.on_variables_client_init_start()

        self.on_variables_client_init()

        self.on_variables_client_adjust_and_request()

        self.on_variables_client_thread_pool()

        self.on_variables_client_futures()

        self.on_variables_client_init_end()

    def get_async(self) -> None:
        """Asynchronously updates the get variables with the latest copy from source."""
        self.on_variables_client_get_start()

        self.on_variables_client_get()

        self.on_variables_client_get_end()

    def set_async(self) -> None:
        """Asynchronously updates source with the set variables."""
        self.on_variables_client_set_start()

        self.on_variables_client_set()

        self.on_variables_client_set_end()

    def set_and_get_async(self) -> None:
        """Asynchronously updates source and gets from source."""
        self.on_variables_client_set_and_get_start()

        self.on_variables_client_set_and_get()

        self.on_variables_client_set_and_get_end()

    def add_async(self, names: List[str], vars: Dict[str, Any]) -> None:
        """Asynchronously adds to source variables."""
        self._names = names
        self._vars = vars

        self.on_variables_client_add_start()

        self.on_variables_client_add()

        self.on_variables_client_add_end()

    def add_and_wait(self, names: List[str], vars: Dict[str, Any]) -> None:
        """Adds the specified variables to the corresponding variables in source
        and waits for the process to complete before continuing."""
        self._client.add_to_variables(names, vars)

    def get_and_wait(self) -> None:
        """Updates the get variables with the latest copy from source
        and waits for the process to complete before continuing."""
        self._copy(self._request())  # type: ignore

    def get_all_and_wait(self) -> None:
        """Updates all the variables with the latest copy from source
        and waits for the process to complete before continuing."""
        self._copy(self._request_all())  # type: ignore

    def set_and_wait(self) -> None:
        """Updates source with the set variables
        and waits for the process to complete before continuing."""
        self._adjust()  # type: ignore

    def _copy(self, new_variables: Dict[str, Any]) -> None:
        """Copies the new variables to the old ones."""
        self._new_variables = new_variables

        self.on_variables_client_start()

        for key in new_variables.keys():
            self._key = key
            var_type = type(new_variables[key])
            if var_type == dict:
                self.on_variables_client_copy_if_dict()
            elif var_type == np.int32 or var_type == np.float32:
                self.on_variables_client_copy_if_int_float()
            elif var_type == tuple:
                self.on_variables_client_copy_if_tuple()
            else:
                NotImplementedError(f"Variable type of {var_type} not implemented.")

        self.on_variables_client_copy_end()