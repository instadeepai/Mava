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

# NOTE (StJohn): I'm still thinking about the structure in general. One of the biggest
# design choices at the moment is how to structure what is passed to the __init__ when
# instantiating vs what is passed to the forward functions. The distinction is due to
# the differences in what VDN and Qmix need.

# Code inspired by PyMARL framework implementation
# https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/qmix.py

"""Mixing for multi-agent RL systems"""

from typing import Dict

import tensorflow as tf
from tensorflow import Tensor

from mava.components.tf.architectures import BaseArchitecture
from mava.components.tf.modules.mixing import BaseMixingModule


class MonotonicMixing(BaseMixingModule):
    """Multi-agent mixing architecture."""

    def __init__(
        self,
        architecture: BaseArchitecture,
        state_shape: tuple,
        hypernet_embed: tuple,
        n_agents: int,
        embed_dim: tuple,
        mixer: str = "vdn",
    ) -> None:
        """Initializes the mixer.
        Args:
            architecture: the BaseArchitecture used.
            mixer: the type of monotonic mixing.
        """
        self._architecture = architecture
        self._state_shape = state_shape  # Defined by the environment
        self._hypernet_embed = hypernet_embed
        self._n_agents = n_agents
        self._embed_dim = embed_dim
        self._mixer = mixer.lower()  # Ensure we accept different cases vDn qMiX etc

    def forward(
        self,
        agent_qs: Dict[str, float],  # Check type
        states: Dict[str, float],  # Check type
        num_hypernet_layers: int = 1,
    ) -> Tensor:

        """Monotonic mixing logic."""
        if self._mixer == "vdn":
            # Not sure if this is the way to simply sum in tf.
            # I'm looking for an equivalent to th.sum(...) in PyTorch.
            return tf.math.accumulate_n(agent_qs)

        elif self._mixer == "qmix":
            # Set up hypernetwork configuration
            if num_hypernet_layers == 1:
                self.hyper_w_1 = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Flatten(input_shape=self._state_shape),
                        tf.keras.layers.Dense(self._embed_dim * self._n_agents),
                    ]
                )
                self.hyper_w_final = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Flatten(input_shape=self._state_shape),
                        tf.keras.layers.Dense(self._state_shape),
                    ]
                )
            elif num_hypernet_layers == 2:
                self.hyper_w_1 = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Flatten(input_shape=self._state_shape),
                        tf.keras.layers.Dense(self._hypernet_embed, activation="relu"),
                        tf.keras.layers.Dense(self._embed_dim * self._n_agents),
                    ]
                )
                self.hyper_w_final = tf.keras.models.Sequential(
                    [
                        tf.keras.layers.Flatten(input_shape=self._state_shape),
                        tf.keras.layers.Dense(self._hypernet_embed, activation="relu"),
                        tf.keras.layers.Dense(self._embed_dim),
                    ]
                )
            elif num_hypernet_layers > 2:
                raise Exception("Sorry >2 hypernet layers is not implemented!")
            else:
                raise Exception("Error setting number of hypernet layers.")

            # State dependent bias for hidden layer

            # # V(s) instead of a bias for the last layers

            # # Forward pass
            # bs = agent_qs.size(0)
            # states = tf.reshape(states, shape=self._state_shape)
            # agent_qs = tf.reshape(agent_qs, shape=(-1, 1))
            # # First layer
            # w1 = tf.abs(self.hyper_w_1(states))

    # # Oxwhirl PyTorch implementation of forward pass
    # def forward(self, agent_qs, states):
    #     bs = agent_qs.size(0)
    #     states = states.reshape(-1, self.state_dim)
    #     agent_qs = agent_qs.view(-1, 1, self.n_agents)
    #     # First layer
    #     w1 = th.abs(self.hyper_w_1(states))
    #     b1 = self.hyper_b_1(states)
    #     w1 = w1.view(-1, self.n_agents, self.embed_dim)
    #     b1 = b1.view(-1, 1, self.embed_dim)
    #     hidden = F.elu(th.bmm(agent_qs, w1) + b1)
    #     # Second layer
    #     w_final = th.abs(self.hyper_w_final(states))
    #     w_final = w_final.view(-1, self.embed_dim, 1)
    #     # State-dependent bias
    #     v = self.V(states).view(-1, 1, 1)
    #     # Compute final output
    #     y = th.bmm(hidden, w_final) + v
    #     # Reshape and return
    #     q_tot = y.view(bs, -1, 1)
    #     return q_tot
