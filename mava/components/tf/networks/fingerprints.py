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

"""Sonnet module that takes two inputs
[observation, fingerprint]"""

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
    ) -> None:
        """Initializes network.
        Args:
            observation_network: ...
        """
        super(ObservationNetworkWithFingerprint, self).__init__()
        self._observation_network = observation_network
        self._flatten_layer = tf.keras.layers.Flatten()

    def __call__(
        self,
        obs: tf.Tensor,
        fingerprint: tf.Tensor,
    ) -> tf.Tensor:

        hidden = self._observation_network(obs)
        flatten = self._flatten_layer(hidden)
        hidden_with_fingerprint = tf.concat([flatten, fingerprint], axis=1)

        return hidden_with_fingerprint
