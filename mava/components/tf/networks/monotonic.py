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

# Code inspired by PyMARL framework implementation
# https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/qmix.py

"""Mixing for multi-agent RL systems"""

from typing import Dict

import sonnet as snt
import tensorflow as tf

from mava.components.tf.networks.hypernetwork import HyperNetwork


class MonotonicMixingNetwork(snt.Module):
    """Multi-agent monotonic mixing architecture.
    This is the component which can be used to add monotonic mixing to an underlying
    agent architecture. It currently supports generalised monotonic mixing using
    hypernetworks (1 or 2 layers) for control of decomposition parameters (QMix)."""

    def __init__(
        self,
        agent_networks: Dict[str, snt.Module],
        n_agents: int,
        name: str = "mixing",
        qmix_hidden_dim: int = 64,
        num_hypernet_layers: int = 2,
        hypernet_hidden_dim: int = 0,
    ) -> None:
        """Initializes the mixer.
        Args:
            state_shape: The state shape as defined by the environment.
            n_agents: The number of agents (i.e. Q-values) to mix.
            qmix_hidden_dim: Mixing layers hidden dimensions.
            num_hypernet_layers: Number of hypernetwork layers. Currently 1 or 2.
            hypernet_hidden_dim: The number of nodes in the hypernetwork hidden
                layer. Relevant for num_hypernet_layers > 1.
        """
        super(MonotonicMixingNetwork, self).__init__(name=name)
        self._agent_networks = agent_networks
        self._n_agents = n_agents
        self._qmix_hidden_dim = qmix_hidden_dim
        self._num_hypernet_layers = num_hypernet_layers
        self._hypernet_hidden_dim = hypernet_hidden_dim

        # Create hypernetwork
        self._hypernetworks = HyperNetwork(
            self._agent_networks,
            self._qmix_hidden_dim,
            self._n_agents,
            self._num_hypernet_layers,
            self._hypernet_hidden_dim,
        )

    def __call__(
        self,
        q_values: tf.Tensor,  # [batch_size, n_agents]
        states: tf.Tensor,  # [batch_size, state_dim]
    ) -> tf.Tensor:
        """Monotonic mixing logic."""

        # Create hypernetwork
        self._hyperparams = self._hypernetworks(states)

        # Extract hypernetwork layers
        # TODO: make more general -> this assumes two layer hypernetwork
        w1 = self._hyperparams["w1"]  # [B, n_agents, qmix_hidden_dim]
        b1 = self._hyperparams["b1"]  # [B, 1, qmix_hidden_dim]
        w2 = self._hyperparams["w2"]  # [B, qmix_hidden_dim, 1]
        b2 = self._hyperparams["b2"]  # [B, 1, 1]

        # ELU -> Exp. linear unit
        hidden = tf.nn.elu(tf.matmul(q_values, w1) + b1)  # [B, 1, qmix_hidden_dim]

        # Qtot: [B, 1, 1]
        return tf.matmul(hidden, w2) + b2
