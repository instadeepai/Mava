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

"""Mixing for multi-agent RL systems"""
from typing import Dict, Union

import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils

from mava import specs as mava_specs
from mava.components.tf.architectures.decentralised import (
    DecentralisedValueActor,
    DecentralisedValueActorCritic,
)
from mava.components.tf.modules.mixing import BaseMixingModule
from mava.components.tf.networks.monotonic import MonotonicMixingNetwork


class MonotonicMixing(BaseMixingModule):
    """Multi-agent monotonic mixing architecture.
    This is the component which can be used to add monotonic mixing to an underlying
    agent architecture. It currently supports generalised monotonic mixing using
    hypernetworks (1 or 2 layers) for control of decomposition parameters (QMix)."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        architecture: Union[DecentralisedValueActor, DecentralisedValueActorCritic],
        qmix_hidden_dim: int = 32,
        num_hypernet_layers: int = 2,
        hypernet_hidden_dim: int = 64,  # Defaults to qmix_hidden_dim
    ) -> None:
        """Initializes the mixer.
        Args:
            architecture: the BaseArchitecture used.
        """
        super(MonotonicMixing, self).__init__()

        assert hasattr(
            architecture, "_n_agents"
        ), "Architecture doesn't have _n_agents."
        self._environment_spec = environment_spec
        self._qmix_hidden_dim = qmix_hidden_dim
        self._num_hypernet_layers = num_hypernet_layers
        self._hypernet_hidden_dim = hypernet_hidden_dim
        self._n_agents = architecture._n_agents
        self._architecture = architecture
        self._agent_networks = self._architecture.create_actor_variables()

    def _create_mixing_layer(self, name: str = "mixing") -> snt.Module:
        """Modify and return system architecture given mixing structure."""
        state_specs = self._environment_spec.get_extra_specs()
        state_specs = state_specs["s_t"]

        q_value_dim = tf.TensorSpec(self._n_agents)

        # Implement method from base class
        self._mixed_network = MonotonicMixingNetwork(
            agent_networks=self._agent_networks,
            n_agents=self._n_agents,
            qmix_hidden_dim=self._qmix_hidden_dim,
            num_hypernet_layers=self._num_hypernet_layers,
            hypernet_hidden_dim=self._hypernet_hidden_dim,
            name=name,
        )

        tf2_utils.create_variables(self._mixed_network, [q_value_dim, state_specs])
        return self._mixed_network

    def create_system(self) -> Dict[str, Dict[str, snt.Module]]:
        # Implement method from base class
        self._agent_networks["mixing"] = self._create_mixing_layer(name="mixing")
        self._agent_networks["target_mixing"] = self._create_mixing_layer(
            name="target_mixing"
        )

        return self._agent_networks
