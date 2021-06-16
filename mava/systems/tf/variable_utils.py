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

"""Variable handling utilities for TensorFlow 2. Adapted from Deepmind's Acme library"""

import copy
from concurrent import futures
from typing import Mapping, Optional, Sequence

import tensorflow as tf
import tree
from acme import core
from acme.tf import utils as tf2_utils
from numpy.core.fromnumeric import var

from mava.systems.tf.variable_sources import VariableSource


class VariableClient:
    """A variable client for updating variables from a remote source."""

    def __init__(
        self,
        client: VariableSource,
        variables: Mapping[str, Sequence[tf.Variable]],
        get_period: int = 1,
        set_period: int = 1,
    ):
        # TODO(dries): Remove this
        self._keys = list(variables.keys())
        # TODO (dries): Remove the deepcopy
        self._variables = tf2_utils.to_numpy(
            variables
        )  # tree.flatten([variables[key] for key in variables.keys()]) # variables #
        self._get_call_counter = 0
        self._set_call_counter = 0
        self._get_update_period = get_period
        self._set_update_period = set_period
        self._client = client
        self._request = lambda: client.get_variables(self._keys)
        import numpy as np

        self._adjust = lambda: client.set_variables(
            self._keys, self._variables
        )  # , tf2_utils.to_numpy(self._variables)  #client.set_variables(self._keys, self._variables)
        # Create a single background thread to fetch variables without necessarily
        # blocking the actor.
        self._executor = futures.ThreadPoolExecutor(max_workers=1)
        self._async_request = lambda: self._executor.submit(self._request)
        self._async_adjust = lambda: self._executor.submit(self._adjust)

        # Initialize this client's future to None to indicate to the `update()`
        # method that there is no pending/running request.
        self._future: Optional[futures.Future] = None

    def get_async(self):
        """Periodically updates the variables with the latest copy from the source.

        This stateful update method keeps track of the number of calls to it and,
        every `update_period` call, sends a request to its server to retrieve the
        latest variables.

        If wait is True, a blocking request is executed. Any active request will be
        cancelled.
        If wait is False, this method makes an asynchronous request for variables
        and returns. Unless the request is immediately fulfilled, the variables are
        only copied _within a subsequent call to_ `update()`, whenever the request
        is fulfilled by the `VariableSource`. If there is an existing fulfilled
        request when this method is called, the resulting variables are immediately
        copied.

        Args:
          wait: if True, executes blocking update.
        """
        # Track the number of calls (we only update periodically).
        if self._get_call_counter < self._get_update_period:
            self._get_call_counter += 1

        period_reached: bool = self._get_call_counter >= self._get_update_period

        # if period_reached and wait:
        #   # Cancel any active request.
        #   self._future: Optional[futures.Future] = None
        #   self.get_and_wait()
        #   self._call_counter = 0
        #   return

        if period_reached and self._future is None:
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            self._future = self._async_request()

        if self._future is not None and self._future.done():
            # The active request is done so copy the result and remove the future.\
            self._copy(self._future.result())
            self._future: Optional[futures.Future] = None
            self._get_call_counter = 0

        return

    def set_async(self, variables):
        # TODO (dries): Fix this so that the variable client and trainer uses the same wieghts. No need to assign again.
        # tree.flatten([variables[key] for key in variables.keys()])

        # Track the number of calls (we only update periodically).
        if self._set_call_counter < self._set_update_period:
            self._set_call_counter += 1

        period_reached: bool = self._set_call_counter >= self._set_update_period

        if period_reached and self._future is None:
            for key in variables.keys():
                self._variables[key] = tf2_utils.to_numpy(variables[key])
            # The update period has been reached and no request has been sent yet, so
            # making an asynchronous request now.
            self._future = self._async_adjust()
            return
        if self._future is not None and self._future.done():
            self._future: Optional[futures.Future] = None
            self._set_call_counter = 0
        return

    def get_and_wait(self):
        """Immediately update and block until we get the result."""
        self._copy(self._request())

    def set_and_wait(self, variables):
        """Immediately update and block until we get the result."""
        # TODO (dries): Fix this so that the variable client and trainer uses the same wieghts. No need to assign again.

        for key in variables.keys():
            self._variables[key] = tf2_utils.to_numpy(variables[key])
        # self._variables = variables # tree.flatten([variables[key] for key in variables.keys()])
        self._adjust()

    def _copy(self, new_variables: Sequence[Sequence[tf.Variable]]):
        """Copies the new variables to the old ones."""

        # new_variables = tree.flatten(new_variables)
        # if len(self._variables) != len(new_variables):
        #     raise ValueError('Length mismatch between old (',len(self._variables),') variables and new (',len(new_variables),').')

        # new_variables = tree.flatten(new_variables)
        if len(self._variables) != len(new_variables):
            raise ValueError(
                "Length mismatch between old ",
                self._variables.keys(),
                " variables and new",
                new_variables.keys(),
                ".",
            )

        # for new, old in zip(new_variables, self._variables):
        #     self._variables[i] = new_variables[i]
        #   old.assign(new)

        for key in new_variables.keys():
            # TODO (dries): Maybe do a old.assign(new) or a copy?
            # if type(self._variables[key]) == dict:
            #     for agent in self._variables[key].keys():
            #         assert self._variables[key][agent].shape == new_variables[key][agent].shape
            # else:
            #     assert self._variables[key].shape == new_variables[key].shape

            if type(new_variables[key]) == dict:
                for agent_key in new_variables[key].keys():
                    self._variables[key][agent_key] = new_variables[key][agent_key]
            else:
                self._variables[key] = new_variables[key]
