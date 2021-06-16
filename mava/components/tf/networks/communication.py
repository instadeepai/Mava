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

from typing import Optional

import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils


class CommunicationNetwork(snt.Module):
    def __init__(
        self,
        obs_in_network: snt.Module,
        comm_in_network: snt.Module,
        core_network: snt.RNNCore,
        action_head: snt.Module,
        comm_head: snt.Module,
        message_size: int,
        name: str = None,
    ):
        super(CommunicationNetwork, self).__init__(name=name)

        self._obs_in_network = obs_in_network
        self._comm_in_network = comm_in_network
        self._core_network = core_network
        self._action_head = action_head
        self._comm_head = comm_head

        self._message_size = message_size

    def initial_state(self, batch_size: int = 1) -> snt.Module:
        # Requires core_network to be of type snt.RNNCore
        return self._core_network.initial_state(batch_size)

    def initial_message(self, batch_size: int = 1) -> snt.Module:
        # TODO Kevin: get output dim of comm_head
        return tf.zeros([batch_size, self._message_size], dtype=tf.float32)

    def __call__(
        self,
        x: snt.Module,
        state: Optional[snt.Module] = None,
        message: Optional[snt.Module] = None,
    ) -> snt.Module:
        if state is None:
            state = self.initial_state(x.shape[0])
        if message is None:
            message = self.initial_message(x.shape[0])

        obs_in = self._obs_in_network(x)
        comm_in = self._comm_in_network(message)

        core_in = tf2_utils.batch_concat([obs_in, comm_in])

        core_out, state = self._core_network(core_in, state)

        action = self._action_head(core_out)
        message = self._comm_head(core_out)

        return (action, message), state
