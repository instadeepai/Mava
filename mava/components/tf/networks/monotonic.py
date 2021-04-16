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
#   - [] Think about how to default values.

# NOTE (StJohn): I'm still thinking about the structure in general. One of the biggest
# design choices at the moment is how to structure what is passed to the __init__ when
# instantiating vs what is passed to the forward functions.

# Code inspired by PyMARL framework implementation
# https://github.com/oxwhirl/pymarl/blob/master/src/modules/mixers/qmix.py

"""Mixing for multi-agent RL systems"""

from typing import Dict

import numpy as np
import sonnet as snt
import tensorflow as tf

from mava.components.tf.networks.hypernetwork import HyperNetwork

class MonotonicNetwork(snt.Module):
    """Multi-agent monotonic mixing architecture.
    This is the component which can be used to add monotonic mixing to an underlying
    agent architecture. It currently supports generalised monotonic mixing using
    hypernetworks (1 or 2 layers) for control of decomposition parameters (QMix)."""

    def __init__(
        self,
        qmix_hidden_dim: int,
        state_dim: int,
        n_agents: int, # TODO Get from architecture
        num_hypernet_layers: int = 2,
        hypernet_hidden_dim: int = 64,
    ) -> None:
        """Initializes the mixer.
        Args:
            architecture: the BaseArchitecture used.
            state_shape: The state shape as defined by the environment.
            n_agents: The number of agents (i.e. Q-values) to mix.
            qmix_hidden_dim: Mixing layers hidden dimensions.
            num_hypernet_layers: Number of hypernetwork layers. Currently 1 or 2.
            hypernet_hidden_dim: The number of nodes in the hypernetwork hidden
                layer. Relevant for num_hypernet_layers > 1.
        """
        self._qmix_hidden_dim = qmix_hidden_dim
        self._state_dim = state_dim
        self._n_agents = n_agents
        self._num_hypernet_layers = num_hypernet_layers
        self._hypernet_hidden_dim = hypernet_hidden_dim

        # Create hypernetwork
        self._hypernetworks = HyperNetwork(
            self._state_dim,
            self._n_agents,
            self._num_hypernet_layers,
            self._hypernet_hidden_dim,
        )

    def __call__(
        self,
        q_values: tf.Tensor,  # Check type
        states: tf.Tensor, # Check type
    ) -> tf.Tensor:
        """Monotonic mixing logic."""
        states = tf.reshape(states, shape=(-1, self._state_dim))
        q_values = tf.reshape(q_values, shape=(-1, 1, self._n_agents))
        episode_num = tf.size(q_values).numpy()  # Get int from 0D tensor length

        self._hyperparams = self._hypernetworks(states)
        self._hyperparams["w1"] = tf.reshape(
            self._hyperparams["w1"], shape=(-1, self._n_agents, self._qmix_hidden_dim)
        )
        self._hyperparams["b1"] = tf.reshape(
            self._hyperparams["b1"], shape=(-1, 1, self._qmix_hidden_dim)
        )
        self._hyperparams["w2"] = tf.reshape(
            self._hyperparams["w2"], shape=(-1, self._qmix_hidden_dim, 1)
        )
        self._hyperparams["b2"] = tf.reshape(self._hyperparams["b2"], shape=(-1, 1, 1))

        # For convenience
        w1 = self._hyperparams["w1"]
        b1 = self._hyperparams["b1"]
        w2 = self._hyperparams["w2"]
        b2 = self._hyperparams["b2"]

        hidden = tf.nn.elu(tf.matmul(q_values, w1) + b1)  # ELU -> Exp. linear unit

        q_tot = tf.matmul(hidden, w2) + b2
        q_tot = tf.reshape(q_tot, shape=(episode_num, -1, 1))
        return q_tot
