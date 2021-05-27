# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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

"""Sonnet module that takes two inputs
[observation, fingerprint]"""
from typing import Optional

import sonnet as snt
import tensorflow as tf


class ObservationNetworkWithFingerprint(snt.Module):
    """Sonnet module that takes two inputs
    [observation, fingerprint] and returns an observation
    embedding and concatenates the fingerprint to the
    embedding. Downstream layers can then be trained
    on the embedding+fingerprint."""

    def __init__(
        self,
        observation_network: snt.Module,
        recurrent_network: Optional[snt.Module] = None,
        output_head: Optional[snt.Module] = None,
    ) -> None:
        """Initializes network.
        Args:
            observation_network: ...
            recurrent_network: ...
            observation_head: ...
        """
        super(ObservationNetworkWithFingerprint, self).__init__()
        self._observation_network = observation_network
        self._flatten_layer = tf.keras.layers.Flatten()
        self._recurrent_network = recurrent_network
        self._output_head = output_head

    def initial_state(self, batch_size: int = 1) -> snt.Module:
        # Requires core_network to be of type snt.RNNCore
        if self._recurrent_network is not None:
            return self._recurrent_network.initial_state(batch_size)
        else:
            return None

    def __call__(
        self, obs: tf.Tensor, fingerprint: tf.Tensor, state: Optional[tf.Tensor] = None
    ) -> tf.Tensor:

        hidden = self._observation_network(obs)
        hidden = self._flatten_layer(hidden)

        if self._recurrent_network is not None:
            if state is None:
                state = self.initial_state(obs.shape[0])
            hidden, state = self._recurrent_network(hidden, state)

        hidden = tf.concat([hidden, fingerprint], axis=1)

        if self._output_head is not None:
            hidden = self._output_head(hidden)

        if state is not None:
            return hidden, state
        else:
            return hidden
