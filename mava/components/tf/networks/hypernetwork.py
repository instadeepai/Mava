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

from typing import Dict

import sonnet as snt
import tensorflow as tf
from tensorflow import Tensor


class HyperNetwork(snt.Module):
    def __init__(
        self,
        agent_networks: Dict[str, snt.Module],
        qmix_hidden_dim: int,  # qmix_hidden_dim
        n_agents: int,
        num_hypernet_layers: int = 2,
        hypernet_hidden_dim: int = 0,  # qmix_hidden_dim
    ):
        """Initializes the mixer.
        Args:
            agent_networks: Networks which produce outputs for mixing network.
            qmix_hidden_dim: Mixing layers hidden dimensions.
                i.e. What size the mixing network takes as input.
            num_hypernet_layers: Number of hypernetwork layers. Currently 1 or 2.
            hypernet_hidden_dim: The number of nodes in the hypernetwork hidden
                layer. Relevant for num_hypernet_layers > 1.
        """
        super(HyperNetwork, self).__init__()
        self._agent_networks = agent_networks
        self._qmix_hidden_dim = qmix_hidden_dim
        self._num_hypernet_layers = num_hypernet_layers
        self._n_agents = n_agents

        # Let the user define the hidden dim but default it to qmix_hidden_dim.
        if hypernet_hidden_dim == 0:
            self._hypernet_hidden_dim = qmix_hidden_dim
        else:
            self._hypernet_hidden_dim = hypernet_hidden_dim

        # Set up hypernetwork configuration
        if self._num_hypernet_layers == 1:
            self.hyper_w1 = snt.nets.MLP(
                output_sizes=[self._qmix_hidden_dim * self._n_agents]
            )
            self.hyper_w2 = snt.nets.MLP(output_sizes=[self._qmix_hidden_dim])

        # Default
        elif self._num_hypernet_layers == 2:
            self.hyper_w1 = snt.nets.MLP(
                output_sizes=[
                    self._hypernet_hidden_dim,
                    self._qmix_hidden_dim * self._n_agents,
                ]
            )
            self.hyper_w2 = snt.nets.MLP(
                output_sizes=[self._hypernet_hidden_dim, self._qmix_hidden_dim]
            )

        # State dependent bias for hidden layer
        self.hyper_b1 = snt.nets.MLP(output_sizes=[self._qmix_hidden_dim])
        self.hyper_b2 = snt.nets.MLP(output_sizes=[self._qmix_hidden_dim, 1])

    def __call__(self, states: Tensor) -> Dict[str, float]:  # [batch_size=B, state_dim]
        w1 = tf.abs(
            self.hyper_w1(states)
        )  # [B, qmix_hidden_dim] = [B, qmix_hidden_dim]
        w1 = tf.reshape(
            w1,
            (-1, self._n_agents, self._qmix_hidden_dim),
        )  # [B, n_agents, qmix_hidden_dim]

        b1 = self.hyper_b1(states)  # [B, qmix_hidden_dim] = [B, qmix_hidden_dim]
        b1 = tf.reshape(b1, [-1, 1, self._qmix_hidden_dim])  # [B, 1, qmix_hidden_dim]

        w2 = tf.abs(self.hyper_w2(states))
        w2 = tf.reshape(
            w2, shape=(-1, self._qmix_hidden_dim, 1)
        )  # [B, qmix_hidden_dim, 1]

        b2 = self.hyper_b2(states)  # [B, 1]
        b2 = tf.reshape(b2, shape=(-1, 1, 1))  # [B, 1, 1]

        hyperparams = {}
        hyperparams["w1"] = w1  # [B, n_agents, qmix_hidden_dim]
        hyperparams["b1"] = b1  # [B, 1, qmix_hidden_dim]
        hyperparams["w2"] = w2  # [B, qmix_hidden_dim]
        hyperparams["b2"] = b2  # [B, 1]

        return hyperparams
