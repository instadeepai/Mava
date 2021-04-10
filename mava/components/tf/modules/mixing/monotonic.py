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

# TODO (StJohn):
#   - [] Complete class for monotonic mixing
#   - [] Generalise Qmixing to allow for different activations,
#          hypernetwork structures etc.
#   - [] Decide on whether to accept an 'args' term or receive each arg individually.

# NOTE (StJohn): I'm still thinking about the structure in general. One of the biggest
# design choices at the moment is how to structure what is passed to the __init__ when
# instantiating vs what is passed to the forward functions. The distinction is due to
# the differences in what VDN and Qmix need.

# Code inspired by PyMARL framework implementation
# https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/qmix.py

"""Mixing for multi-agent RL systems"""

from typing import Tuple

import tensorflow as tf
from tensorflow import Tensor

from mava.components.tf.architectures import BaseArchitecture
from mava.components.tf.modules.mixing import BaseMixingModule


class MonotonicMixing(BaseMixingModule):
    """Multi-agent mixing architecture."""

    def __init__(
        self,
        architecture: BaseArchitecture,
        state_dim: int,
        hypernet_embed: Tuple,
        n_agents: int,
        hypernet_hidden_dim: int,
        num_hypernet_layers: int,
        qmix_hidden_dim: Tuple,
        mixer: str = "vdn",
    ) -> None:
        """Initializes the mixer.
        Args:
            architecture: the BaseArchitecture used.
            mixer: the type of monotonic mixing.
        """
        self._architecture = architecture
        self._state_dim = state_dim  # Defined by the environment
        self._hypernet_embed = hypernet_embed
        self._n_agents = n_agents
        self._hypernet_hidden_dim = hypernet_hidden_dim
        self._num_hypernet_layers = num_hypernet_layers
        self._qmix_hidden_dim = qmix_hidden_dim
        self._mixer = mixer.lower()  # Ensure we accept different cases vDn qMiX etc

        # Set up hypernetwork configuration
        if self._mixer == "qmix":
            if num_hypernet_layers == 1:
                self.hyper_w1 = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Flatten(input_shape=self._state_dim),
                        tf.keras.layers.Dense(self._qmix_hidden_dim * self._n_agents),
                    ]
                )
                self.hyper_w2 = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Flatten(input_shape=self._state_dim),
                        tf.keras.layers.Dense(self._qmix_hidden_dim),
                    ]
                )
            elif num_hypernet_layers == 2:
                self.hyper_w1 = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Flatten(input_shape=self._state_dim),
                        tf.keras.layers.Dense(
                            self._hypernet_hidden_dim, activation="relu"
                        ),
                        tf.keras.layers.Dense(self._qmix_hidden_dim * self._n_agents),
                    ]
                )
                self.hyper_w2 = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Flatten(input_shape=self._state_dim),
                        tf.keras.layers.Dense(
                            self._hypernet_hidden_dim, activation="relu"
                        ),
                        tf.keras.layers.Dense(self._qmix_hidden_dim),
                    ]
                )
            elif num_hypernet_layers > 2:
                raise Exception("Sorry >2 hypernet layers is not implemented!")
            else:
                raise Exception("Error setting number of hypernet layers.")

            # State dependent bias for hidden layer
            self.hyper_b1 = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Flatten(input_shape=self._state_dim),
                    tf.keras.layers.Dense(self._qmix_hidden_dim),
                ]
            )
            self.hyper_b2 = tf.keras.models.Sequential(
                [
                    tf.keras.layers.Flatten(input_shape=self._state_dim),
                    tf.keras.layers.Dense(self._qmix_hidden_dim, activation="relu"),
                    tf.keras.layers.Dense(1),
                ]
            )

    def forward(
        self,
        q_values: Tensor,  # Check type
        states: Tensor,  # Check type
    ) -> Tensor:

        """Monotonic mixing logic."""
        if self._mixer == "vdn":
            # Not sure if this is the way to simply sum in tf.
            # I'm looking for an equivalent to th.sum(...) in PyTorch.
            return self._vdn_forward(q_values)

        elif self._mixer == "qmix":
            return self._qmix_forward(q_values, states)

    def _vdn_forward(self, q_values: Tensor) -> Tensor:
        """Helper function to implement forward pass logic for VDN network."""
        return tf.math.accumulate_n(q_values)

    def _qmix_forward(
        self,
        q_values: Tensor,  # Check type
        states: Tensor,  # Check type
    ) -> Tensor:
        """Helper function to implement forward pass logic for Qmix network."""
        # Forward pass
        episode_num = tf.size(q_values).numpy()  # Get int from 0D tensor length
        states = tf.reshape(states, shape=(-1, self._state_dim))
        q_values = tf.reshape(q_values, shape=(-1, 1, self._n_agents))

        # First layer
        w1 = tf.abs(self.hyper_w1(states))
        b1 = self.hyper_w1(states)
        w1 = tf.reshape(w1, shape=(-1, self._n_agents, self._qmix_hidden_dim))
        b1 = tf.reshape(b1, shape=(-1, 1, self._qmix_hidden_dim))
        hidden = tf.nn.elu(
            tf.linalg.matmul(q_values, w1) + b1
        )  # ELU -> Exp. linear unit

        # Second layer
        w2 = tf.abs(self.hyper_w2(states))
        b2 = self.hyper_b2(states)
        w2 = tf.reshape(w2, shape=(-1, self._qmix_hidden_dim, 1))
        b2 = tf.reshape(b2, shape=(-1, 1, 1))

        q_tot = tf.linalg.matmul(hidden, w2) + b2
        q_tot = tf.reshape(q_tot, shape=(episode_num, -1, 1))
        return q_tot
