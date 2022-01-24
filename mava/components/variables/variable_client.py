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

import abc

from concurrent import futures
from typing import Any, Dict, List, Optional

from mava.callbacks import Callback
from mava.core import SystemVariableClient
from mava.components.variables.variable_server import VariableServer
from mava.utils.sort_utils import sort_str_num


class VariableClient(Callback):
    """A variable client for updating variables from a remote source."""

    def __init__(
        self,
        server: VariableServer,
        variables: Dict[str, Any],
        get_keys: List[str] = None,
        set_keys: List[str] = None,
        update_period: int = 1,
    ) -> None:

        self._server = server
        self._variables = variables
        self._get_keys = get_keys
        self._set_keys = set_keys
        self._update_period = update_period

    def on_variables_client_init(self, client: SystemVariableClient) -> None:
        """Initialise the variable server."""
        client._all_keys = sort_str_num(list(self._variables.keys()))
        client._get_keys = (
            self._get_keys if self._get_keys is not None else self._all_keys
        )
        client._set_keys = (
            self._set_keys if self._set_keys is not None else self._all_keys
        )
        client._get_call_counter = 0
        client._set_call_counter = 0
        client._set_get_call_counter = 0
        client._request = lambda: self._server.get_variables(client._get_keys)
        client._request_all = lambda: self._server.get_variables(client._all_keys)

    @abc.abstractmethod
    def on_variables_client_adjust_and_request(
        self, client: SystemVariableClient
    ) -> None:
        """[summary]"""

    def on_variables_client_thread_pool(self, client: SystemVariableClient) -> None:

        # Create a single background thread to fetch variables without necessarily
        # blocking the actor.
        client._executor = futures.ThreadPoolExecutor(max_workers=1)
        client._async_add_buffer: Dict[str, Any] = {}  # type: ignore
        client._async_request = lambda: client._executor.submit(client._request)
        client._async_adjust = lambda: client._executor.submit(client._adjust)
        client._async_adjust_and_request = lambda: client._executor.submit(
            client._adjust_and_request
        )
        client._async_add = lambda names, vars: client._executor.submit(  # type: ignore
            client._add(names, vars)  # type: ignore
        )

    def on_variables_client_futures(self, client: SystemVariableClient) -> None:
        # Initialize this client's future to None to indicate to the `update()`
        # method that there is no pending/running request.
        # TODO (Arnu): look into type errors when using types with variable client class
        client._get_future: Optional[futures.Future] = None  # type: ignore
        client._set_future: Optional[futures.Future] = None  # type: ignore
        client._set_get_future: Optional[futures.Future] = None  # type: ignore
        client._add_future: Optional[futures.Future] = None  # type: ignore

    def on_variables_client_get(self, client: SystemVariableClient) -> None:
        """Asynchronously updates the get variables with the latest copy from server."""
        # Track the number of calls (we only update periodically).
        if client._get_call_counter < self._update_period:
            client._get_call_counter += 1

        period_reached: bool = client._get_call_counter >= client._update_period

        if period_reached and client._get_future is None:
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            client._get_future = client._async_request()  # type: ignore
            client._get_call_counter = 0

        if client._get_future is not None and client._get_future.done():
            # The active request is done so copy the result and remove the future.\
            client._copy(client._get_future.result())
            client._get_future = None

    def on_variables_client_set(self, client: SystemVariableClient) -> None:
        """Asynchronously updates source with the set variables."""
        # Track the number of calls (we only update periodically).
        if client._set_call_counter < self._update_period:
            client._set_call_counter += 1

        period_reached: bool = client._set_call_counter >= client._update_period

        if period_reached and client._set_future is None:  # type: ignore
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            client._set_future = client._async_adjust()  # type: ignore
            client._set_call_counter = 0
            return
        if client._set_future is not None and client._set_future.done():
            client._set_future = None  # type: ignore

    def on_variables_client_set_and_get(self, client: SystemVariableClient) -> None:
        """Asynchronously updates source and gets from source."""
        # Track the number of calls (we only update periodically).
        if client._set_get_call_counter < self._update_period:
            client._set_get_call_counter += 1
        period_reached: bool = client._set_get_call_counter >= self._update_period

        if period_reached and client._set_get_future is None:  # type: ignore
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            client._set_get_future = client._async_adjust_and_request()  # type: ignore
            client._set_get_call_counter = 0
            return
        if client._set_get_future is not None and client._set_get_future.done():
            client._set_get_future = None  # type: ignore

    def on_variables_client_add(self, client: SystemVariableClient) -> None:
        """Asynchronously adds to source variables."""
        if client._add_future is not None and client._add_future.done():
            client._add_future = None

        if client._add_future is None:
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            if not client._async_add_buffer:
                client._add_future = client._async_add(client._names, client._vars)  # type: ignore
            else:
                for name in client._names:
                    client._async_add_buffer[name] += client._vars[name]
                client._add_future = client._async_add(  # type: ignore
                    client._names, client._async_add_buffer
                )  # type: ignore
                client._async_add_buffer = {}
            return
        else:
            # The trainers is going to fast to keep up! Adding
            # all the values up and only writing them when the
            # process is ready.
            if client._async_add_buffer:
                for name in client._names:
                    client._async_add_buffer[name] += client._vars[name]
            else:
                for name in client._names:
                    client._async_add_buffer[name] = client._vars[name]

    @abc.abstractmethod
    def on_variables_client_copy_if_dict(self, client: SystemVariableClient) -> None:
        """[summary]"""

    @abc.abstractmethod
    def on_variables_client_copy_if_int_float(
        self, client: SystemVariableClient
    ) -> None:
        """[summary]"""

    @abc.abstractmethod
    def on_variables_client_copy_if_tuple(self, client: SystemVariableClient) -> None:
        """[summary]"""
