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
        state_dim: int,
        n_agents: int,
        qmix_hidden_dim: int,
        num_hypernet_layers: int = 2,
        hypernet_hidden_dim: int = 2,
    ):
        """Initializes the mixer.
        Args:
            state_dim: The state dimension as defined by the environment.
            n_agents: The number of agents (i.e. Q-values) to mix.
            qmix_hidden_dim: Mixing layers hidden dimensions.
            num_hypernet_layers: Number of hypernetwork layers. Currently 1 or 2.
            hypernet_hidden_dim: The number of nodes in the hypernetwork hidden
                layer. Relevant for num_hypernet_layers > 1.
        """
        super(HyperNetwork, self).__init__()

        self._state_dim = state_dim
        self._n_agents = state_dim
        self._num_hypernet_layers = num_hypernet_layers
        self._hypernet_hidden_dim = hypernet_hidden_dim
        self._qmix_hidden_dim = qmix_hidden_dim

        # Set up hypernetwork configuration
        if self._num_hypernet_layers == 1:
            self.hyper_w1 = snt.nets.MLP(
                output_sizes=[self._qmix_hidden_dim * self._n_agents]
            )
            self.hyper_w2 = snt.nets.MLP(output_sizes=[self._qmix_hidden_dim])
            # self.hyper_w1 = snt.Sequential(
            #     [
            #         snt.Flatten(preserve_dims=1),
            #         snt.Linear(self._qmix_hidden_dim * self._n_agents),
            #     ]
            # )
            # self.hyper_w2 = snt.Sequential(
            #     [
            #         snt.Flatten(preserve_dims=1),
            #         snt.Linear(self._qmix_hidden_dim),
            #     ]
            # )
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
            # self.hyper_w1 = snt.Sequential(
            #     [
            #         snt.Flatten(preserve_dims=1),
            #         snt.Linear(self._hypernet_hidden_dim),
            #         tf.nn.relu(),
            #         snt.Linear(self._qmix_hidden_dim * self._n_agents),
            #     ]
            # )
            # self.hyper_w2 = snt.Sequential(
            #     [
            #         snt.Flatten(preserve_dims=1),
            #         snt.Linear(self._hypernet_hidden_dim),
            #         tf.nn.relu(),
            #         snt.Linear(self._qmix_hidden_dim),
            #     ]
            # )
        elif self._num_hypernet_layers > 2:
            raise Exception(
                "Sorry >2 hypernet layers is not implemented!",
                self._num_hypernet_layers,
            )
        else:
            raise Exception("Error setting number of hypernet layers.")

        # State dependent bias for hidden layer
        self.hyper_b1 = snt.nets.MLP(output_sizes=[self._qmix_hidden_dim])
        self.hyper_b2 = snt.nets.MLP(output_sizes=[self._qmix_hidden_dim, 1])
        # self.hyper_b1 = snt.Sequential(
        #     [
        #         snt.Flatten(preserve_dims=1),
        #         snt.Linear(self._qmix_hidden_dim),
        #     ]
        # )
        # self.hyper_b2 = snt.Sequential(
        #     [
        #         snt.Flatten(preserve_dims=1),
        #         snt.Linear(self._qmix_hidden_dim),
        #         tf.nn.relu(),
        #         snt.Linear(1),
        #     ]
        # )

    def __call__(self, states: Tensor) -> Dict[str, float]:
        hyperparams = {}
        hyperparams["w1"] = tf.abs(self.hyper_w1(states))
        hyperparams["b1"] = self.hyper_w1(states)
        hyperparams["w2"] = tf.abs(self.hyper_w2(states))
        hyperparams["b2"] = self.hyper_b2(states)

        return hyperparams
