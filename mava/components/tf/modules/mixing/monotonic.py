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

"""Mixing for multi-agent RL systems"""

from typing import Dict, Tuple

import numpy as np
import sonnet as snt

from mava.components.tf.architectures import BaseArchitecture
from mava.components.tf.modules.mixing import BaseMixingModule
from mava.components.tf.networks.monotonic import MonotonicMixingNetwork


class MonotonicMixing(BaseMixingModule):
    """Multi-agent monotonic mixing architecture.
    This is the component which can be used to add monotonic mixing to an underlying
    agent architecture. It currently supports generalised monotonic mixing using
    hypernetworks (1 or 2 layers) for control of decomposition parameters (QMix)."""

    def __init__(
        self,
        architecture: BaseArchitecture,
        state_shape: Tuple,
        n_agents: int,  # TODO Get this from architecture
        qmix_hidden_dim: int,
        num_hypernet_layers: int = 2,
        hypernet_hidden_dim: int = 2,
    ) -> None:
        """Initializes the mixer.
        Args:
            architecture: the BaseArchitecture used.
        """
        super(MonotonicMixing, self).__init__()

        self._architecture = architecture

        self._state_dim = int(np.prod(state_shape))  # Defined by the environment
        self._n_agents = n_agents
        self._qmix_hidden_dim = qmix_hidden_dim
        self._num_hypernet_layers = num_hypernet_layers
        self._hypernet_hidden_dim = hypernet_hidden_dim

    def _create_mixing_layer(self) -> snt.Module:
        """Modify and return system architecture given mixing structure."""
        # Implement method from base class
        self._mixed_network = MonotonicMixingNetwork(
            self._architecture,
            self._qmix_hidden_dim,
            self._state_dim,
            self._n_agents,
            self._num_hypernet_layers,
            self._hypernet_hidden_dim,
        )
        return self._mixed_network

    def create_system(self) -> Dict[str, Dict[str, snt.Module]]:
        # Implement method from base class
        networks = self._architecture.create_actor_variables()
        networks["mixing"] = self._create_mixing_layer()
        return networks
