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

"""Parameter client for Jax system. Adapted from Deepmind's Acme library"""

from concurrent import futures
from typing import Any, Dict, List, Optional, Union

import jax
import numpy as np

from mava.systems.jax.parameter_server import ParameterServer
from mava.utils.sort_utils import sort_str_num


class ParameterClient:
    """A parameter client for updating parameters from a remote server."""

    def __init__(
        self,
        client: ParameterServer,
        parameters: Dict[str, Any],
        get_keys: List[str] = None,
        set_keys: List[str] = None,
        update_period: int = 1,
        devices: Dict[str, Optional[Union[str, jax.xla.Device]]] = {},
    ):
        """Initialise the parameter server."""
        self._all_keys = sort_str_num(list(parameters.keys()))
        # TODO (dries): Is the below change correct?
        self._get_keys = get_keys if get_keys is not None else []
        self._set_keys = set_keys if set_keys is not None else []
        self._parameters: Dict[str, Any] = parameters
        self._get_call_counter = 0
        self._set_call_counter = 0
        self._set_get_call_counter = 0
        self._update_period = update_period
        self._client = client
        self._devices = devices

        # note below it is assumed that if one device is specified with a string
        # they all are - need to test this works
        # TODO: (Dries/Arnu): check this
        if len(self._devices) and isinstance(list(self._devices.values())[0], str):
            for key, device in self._devices.items():
                self._devices[key] = jax.devices(device)[0]  # type: ignore

        self._request = lambda: client.get_parameters(self._get_keys)
        self._request_all = lambda: client.get_parameters(self._all_keys)

        self._adjust = lambda: client.set_parameters(
            {key: self._parameters[key] for key in self._set_keys},
        )

        self._add = lambda params: client.add_to_parameters(params)

        # Create a single background thread to fetch parameters without necessarily
        # blocking the actor.
        self._executor = futures.ThreadPoolExecutor(max_workers=1)
        self._async_add_buffer: Dict[str, Any] = {}
        self._async_request = lambda: self._executor.submit(self._request)
        self._async_adjust = lambda: self._executor.submit(self._adjust)
        self._async_adjust_and_request = lambda: self._executor.submit(
            self._adjust_and_request
        )
        self._async_add: Any = lambda params: self._executor.submit(
            self._add(params)  # type: ignore
        )

        # Initialize this client's future to None to indicate to the `update()`
        # method that there is no pending/running request.
        self._get_future: Optional[futures.Future] = None
        self._set_future: Optional[futures.Future] = None
        self._set_get_future: Optional[futures.Future] = None
        self._add_future: Optional[futures.Future] = None

    def _adjust_and_request(self) -> None:
        self._client.set_parameters(
            {key: self._parameters[key] for key in self._set_keys},
        )
        self._copy(self._client.get_parameters(self._get_keys))

    def get_async(self) -> None:
        """Asynchronously updates the parameters with the latest copy from server."""

        # Track the number of calls (we only update periodically).
        if self._get_call_counter < self._update_period:
            self._get_call_counter += 1

        period_reached: bool = self._get_call_counter >= self._update_period
        if period_reached and self._get_future is None:
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            self._get_future = self._async_request()
            self._get_call_counter = 0

        if self._get_future is not None and self._get_future.done():
            # The active request is done so copy the result and remove the future.\
            self._copy(self._get_future.result())
            self._get_future = None

    def set_async(self) -> None:
        """Asynchronously updates server with the set parameters."""
        # Track the number of calls (we only update periodically).
        if self._set_call_counter < self._update_period:
            self._set_call_counter += 1

        period_reached: bool = self._set_call_counter >= self._update_period

        if period_reached and self._set_future is None:
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            self._set_future = self._async_adjust()
            self._set_call_counter = 0
            return
        if self._set_future is not None and self._set_future.done():
            self._set_future = None

    def set_and_get_async(self) -> None:
        """Asynchronously updates server and gets from server."""
        # Track the number of calls (we only update periodically).
        if self._set_get_call_counter < self._update_period:
            self._set_get_call_counter += 1
        period_reached: bool = self._set_get_call_counter >= self._update_period

        if period_reached and self._set_get_future is None:
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            self._set_get_future = self._async_adjust_and_request()
            self._set_get_call_counter = 0
            return
        if self._set_get_future is not None and self._set_get_future.done():
            self._set_get_future = None

    def add_async(self, params: Dict[str, Any]) -> None:
        """Asynchronously adds to server parameters."""
        if self._add_future is not None and self._add_future.done():
            self._add_future = None
        names = params.keys()
        if self._add_future is None:
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            if not self._async_add_buffer:
                self._add_future = self._async_add(params)
            else:
                for name in names:
                    self._async_add_buffer[name] += params[name]

                # raise NotImplementedError("Is the below line correct?")
                self._add_future = self._async_add(
                    params.keys(), self._async_add_buffer
                )
                self._async_add_buffer = {}
            return
        else:
            # The trainers is going to fast to keep up! Adding
            # all the values up and only writing them when the
            # process is ready.
            if self._async_add_buffer:
                for name in names:
                    self._async_add_buffer[name] += params[name]
            else:
                for name in names:
                    self._async_add_buffer[name] = params[name]

    def add_and_wait(self, params: Dict[str, Any]) -> None:
        """Adds the specified parameters to the corresponding parameters in server \
        and waits for the process to complete before continuing."""
        self._client.add_to_parameters(params)

    def get_and_wait(self) -> None:
        """Updates the get parameters with the latest copy from server \
        and waits for the process to complete before continuing."""
        self._copy(self._request())

    def get_all_and_wait(self) -> None:
        """Updates all the parameters with the latest copy from server \
        and waits for the process to complete before continuing."""
        self._copy(self._request_all())

    def set_and_wait(self) -> None:
        """Updates server with the set parameters \
        and waits for the process to complete before continuing."""
        self._adjust()

    # TODO(Dries/Arnu): this needs a bit of a cleanup
    def _copy(self, new_parameters: Dict[str, Any]) -> None:
        """Copies the new parameters to the old ones."""
        for key in new_parameters.keys():
            if isinstance(new_parameters[key], dict):
                for type1_key in new_parameters[key].keys():
                    for type2_key in self._parameters[key][type1_key].keys():
                        if self._devices:
                            # Move variables to a proper device.
                            # self._parameters[key][type1_key][
                            #     type2_key
                            # ] = jax.device_put(  # type: ignore
                            #     new_parameters[key][type1_key],
                            #     self._devices[key][type1_key],
                            # )
                            raise NotImplementedError(
                                "Support for devices"
                                + "have not been implemented"
                                + "yet in the parameter client."
                            )
                        else:
                            self._parameters[key][type1_key][
                                type2_key
                            ] = new_parameters[key][type1_key][type2_key]
            elif isinstance(new_parameters[key], np.ndarray):
                if self._devices:
                    self._parameters[key] = jax.device_put(
                        new_parameters[key], self._devices[key]  # type: ignore
                    )
                else:
                    # Note (dries): These in-place operators are used instead
                    # of direct assignment to not lose reference to the numpy
                    # array.

                    self._parameters[key] *= 0
                    # Remove last dim of numpy array if needed
                    if new_parameters[key].shape != self._parameters[key].shape:
                        self._parameters[key] += new_parameters[key][0]
                    else:
                        self._parameters[key] += new_parameters[key]

            elif isinstance(new_parameters[key], tuple):
                for i in range(len(self._parameters[key])):
                    if self._devices:
                        self._parameters[key][i] = jax.device_put(
                            new_parameters[key][i],
                            self._devices[key][i],  # type: ignore
                        )
                    else:
                        self._parameters[key][i] = new_parameters[key][i]
            else:
                NotImplementedError(
                    f"Parameter type of {type(new_parameters[key])} not implemented."
                )
