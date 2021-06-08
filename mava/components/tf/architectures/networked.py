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

import copy
from typing import Dict, List, Tuple

import sonnet as snt
import tensorflow as tf
from acme import specs as acme_specs

from mava import specs as mava_specs
from mava.components.tf.architectures.decentralised import (
    DecentralisedPolicyActor,
    DecentralisedQValueActorCritic,
)


class NetworkedPolicyActor(DecentralisedPolicyActor):
    """Networked multi-agent actor critic architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        network_spec: Dict[str, List[str]],
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
                actor_obs_shape.insert(0, len(self._network_spec[agent]))
                actor_obs_specs[agent] = tf.TensorSpec(
                    shape=actor_obs_shape,
                    dtype=tf.dtypes.float32,
                )
        return actor_obs_specs


class NetworkedQValueCritic(DecentralisedQValueActorCritic):
    """Centralised multi-agent actor critic architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        network_spec: Dict[str, List[str]],
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        shared_weights: bool = True,
    ):
        super().__init__(
            environment_spec=environment_spec,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            shared_weights=shared_weights,
        )

        self._network_spec = network_spec

        if self._shared_weights:
            raise Exception(
                "Networked architectures currently do not support weight sharing."
            )

    def _get_critic_specs(
        self,
    ) -> Tuple[Dict[str, acme_specs.Array], Dict[str, acme_specs.Array]]:
        critic_obs_specs: Dict[str, acme_specs.Array] = {}
        critic_act_specs: Dict[str, acme_specs.Array] = {}

        agents_by_type = self._env_spec.get_agents_by_type()

        for agent_type, agents in agents_by_type.items():
            for agent in agents:
                critic_obs_shape = list(copy.copy(self._embed_specs[agent].shape))
                critic_act_shape = list(
                    copy.copy(self._agent_specs[agent].actions.shape)
                )
                critic_obs_shape.insert(0, len(self._network_spec[agent]))
                critic_obs_specs[agent] = tf.TensorSpec(
                    shape=critic_obs_shape,
                    dtype=tf.dtypes.float32,
                )
                critic_act_shape.insert(0, len(self._network_spec[agent]))
                critic_act_specs[agent] = tf.TensorSpec(
                    shape=critic_act_shape,
                    dtype=tf.dtypes.float32,
                )

        return critic_obs_specs, critic_act_specs
