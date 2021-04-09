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

"""Architectures using networked agents for multi-agent RL systems"""


# TODO (Arnu): complete network architecture
from typing import Dict, List

import numpy as np
import sonnet as snt
from acme import specs as acme_specs

from mava import specs
from mava.components.tf.architectures.decentralised import DecentralisedActor


def fully_connected_network_spec(agents: List[str]) -> Dict[str, np.ndarray]:
    """Creates network spec for fully connected agents"""
    network_spec: Dict[str, np.ndarray] = {}
    for agent in agents:
        network_spec[agent] = np.ones((len(agents),))
    return network_spec


class NetworkedSystem:
    """Networked system"""

    def _create_networked_spec(self, agent_key: str) -> Dict[str, acme_specs.Array]:
        """Create network structure specifying connection between agents"""


class NetworkedActor(DecentralisedActor, NetworkedSystem):
    """Networked multi-agent actor critic architecture."""

    def __init__(
        self,
        network_spec: Dict[str, np.ndarray],
        environment_spec: specs.MAEnvironmentSpec,
        policy_networks: Dict[str, snt.Module],
        observation_networks: Dict[str, snt.Module],
        behavior_networks: Dict[str, snt.Module],
        shared_weights: bool = True,
    ):
        super().__init__(
            environment_spec=environment_spec,
            policy_networks=policy_networks,
            observation_networks=observation_networks,
            behavior_networks=behavior_networks,
            shared_weights=shared_weights,
        )

        self._network_spec = network_spec

    def _create_networked_spec(self, agent_key: str) -> Dict[str, acme_specs.Array]:
        """Create network structure specifying connection between agents"""
