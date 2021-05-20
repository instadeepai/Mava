# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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

from typing import Dict, List

import numpy as np
import sonnet as snt
import tensorflow as tf
from acme import specs as acme_specs

from mava import specs
from mava.components.tf.architectures.decentralised import DecentralisedPolicyActor


def fully_connected_network_spec(
    agents_by_type: Dict[str, List[str]]
) -> Dict[str, np.ndarray]:
    """Creates network spec for fully connected agents by agent type"""
    network_spec: Dict[str, np.ndarray] = {}
    for agent_type, agents in agents_by_type.items():
        for agent in agents:
            network_spec[agent] = np.ones((len(agents),))
    return network_spec


class NetworkedPolicyActor(DecentralisedPolicyActor):
    """Networked multi-agent actor critic architecture."""

    def __init__(
        self,
        network_spec: Dict[str, np.ndarray],
        environment_spec: specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
    ):
        super().__init__(
            environment_spec=environment_spec,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            shared_weights=shared_weights,
        )

        self._network_spec = network_spec

        if self._shared_weights:
            raise Exception(
                "Networked architectures currently do not support weight sharing."
            )

    def _get_actor_spec(self, agent_key: str) -> Dict[str, acme_specs.Array]:
        """Create network structure specifying connection between agents"""
        actor_obs_specs: Dict[str, acme_specs.Array] = {}

        agents_by_type = self._env_spec.get_agents_by_type()

        for agent_type, agents in agents_by_type.items():
            actor_obs_shape = list(
                copy.copy(
                    self._agent_type_specs[agent_type].observations.observation.shape
                )
            )
            for agent in agents:
                actor_obs_shape.insert(0, np.sum(self._network_spec[agent]))
                actor_obs_specs[agent] = tf.TensorSpec(
                    shape=actor_obs_shape,
                    dtype=tf.dtypes.float32,
                )
        return actor_obs_specs
