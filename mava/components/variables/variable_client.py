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

    def on_variables_client_init(self) -> None:
        """Initialise the variable server."""
        self._all_keys = sort_str_num(list(self._variables.keys()))
        self._get_keys = (
            self._get_keys if self._get_keys is not None else self._all_keys
        )
        self._set_keys = (
            self._set_keys if self._set_keys is not None else self._all_keys
        )
        self._get_call_counter = 0
        self._set_call_counter = 0
        self._set_get_call_counter = 0
        self._request = lambda: self._server.get_variables(self._get_keys)
        self._request_all = lambda: self._server.get_variables(self._all_keys)

    @abc.abstractmethod
    def on_variables_client_adjust_and_request(self) -> None:
        """[summary]"""

    def on_variables_client_thread_pool(self) -> None:

        # Create a single background thread to fetch variables without necessarily
        # blocking the actor.
        self._executor = futures.ThreadPoolExecutor(max_workers=1)
        self._async_add_buffer: Dict[str, Any] = {}
        self._async_request = lambda: self._executor.submit(self._request)
        self._async_adjust = lambda: self._executor.submit(self._adjust)
        self._async_adjust_and_request = lambda: self._executor.submit(
            self._adjust_and_request
        )
        self._async_add = lambda names, vars: self._executor.submit(  # type: ignore
            self._add(names, vars)  # type: ignore
        )

    def on_variables_client_futures(self) -> None:
        # Initialize this client's future to None to indicate to the `update()`
        # method that there is no pending/running request.
        self._get_future: Optional[futures.Future] = None
        self._set_future: Optional[futures.Future] = None
        self._set_get_future: Optional[futures.Future] = None
        self._add_future: Optional[futures.Future] = None

    def on_variables_client_get(self) -> None:
        """Asynchronously updates the get variables with the latest copy from server."""
        # Track the number of calls (we only update periodically).
        if self._get_call_counter < self._update_period:
            self._get_call_counter += 1

        period_reached: bool = self._get_call_counter >= self._update_period

        if period_reached and self._get_future is None:
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            self._get_future = self._async_request()  # type: ignore
            self._get_call_counter = 0

        if self._get_future is not None and self._get_future.done():
            # The active request is done so copy the result and remove the future.\
            self._copy(self._get_future.result())
            self._get_future = None

    def on_variables_client_set(self) -> None:
        """Asynchronously updates source with the set variables."""
        # Track the number of calls (we only update periodically).
        if self._set_call_counter < self._update_period:
            self._set_call_counter += 1

        period_reached: bool = self._set_call_counter >= self._update_period

        if period_reached and self._set_future is None:  # type: ignore
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            self._set_future = self._async_adjust()  # type: ignore
            self._set_call_counter = 0
            return
        if self._set_future is not None and self._set_future.done():
            self._set_future = None  # type: ignore

    def on_variables_client_set_and_get(self) -> None:
        """Asynchronously updates source and gets from source."""
        # Track the number of calls (we only update periodically).
        if self._set_get_call_counter < self._update_period:
            self._set_get_call_counter += 1
        period_reached: bool = self._set_get_call_counter >= self._update_period

        if period_reached and self._set_get_future is None:  # type: ignore
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            self._set_get_future = self._async_adjust_and_request()  # type: ignore
            self._set_get_call_counter = 0
            return
        if self._set_get_future is not None and self._set_get_future.done():
            self._set_get_future = None  # type: ignore

    def on_variables_client_add(self) -> None:
        """Asynchronously adds to source variables."""
        if self._add_future is not None and self._add_future.done():
            self._add_future = None

        if self._add_future is None:
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            if not self._async_add_buffer:
                self._add_future = self._async_add(self._names, self._vars)  # type: ignore
            else:
                for name in self._names:
                    self._async_add_buffer[name] += self._vars[name]
                self._add_future = self._async_add(  # type: ignore
                    self._names, self._async_add_buffer
                )  # type: ignore
                self._async_add_buffer = {}
            return
        else:
            # The trainers is going to fast to keep up! Adding
            # all the values up and only writing them when the
            # process is ready.
            if self._async_add_buffer:
                for name in self._names:
                    self._async_add_buffer[name] += self._vars[name]
            else:
                for name in self._names:
                    self._async_add_buffer[name] = self._vars[name]

    @abc.abstractmethod
    def on_variables_client_copy_if_dict(self) -> None:
        """[summary]"""

    @abc.abstractmethod
    def on_variables_client_copy_if_int_float(self) -> None:
        """[summary]"""

    @abc.abstractmethod
    def on_variables_client_copy_if_tuple(self) -> None:
        """[summary]"""
