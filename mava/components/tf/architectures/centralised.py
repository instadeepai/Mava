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

"""Commonly used centralised architectures for multi-agent RL systems"""

import copy
from typing import Dict, Tuple

import sonnet as snt
import tensorflow as tf
from acme import specs as acme_specs

from mava import specs as mava_specs
from mava.components.tf.architectures.decentralised import DecentralisedActorCritic


class CentralisedActorCritic(DecentralisedActorCritic):
    """Centralised multi-agent actor critic architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        observation_networks: Dict[str, snt.Module],
        behavior_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
    ):
        super().__init__(
            environment_spec=environment_spec,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            observation_networks=observation_networks,
            behavior_networks=behavior_networks,
            shared_weights=shared_weights,
        )

    def _get_critic_specs(
        self,
    ) -> Tuple[Dict[str, acme_specs.Array], Dict[str, acme_specs.Array]]:
        obs_specs_per_type: Dict[str, acme_specs.Array] = {}
        action_specs_per_type: Dict[str, acme_specs.Array] = {}

        agents_by_type = self._env_spec.get_agents_by_type()

        # Create one critic per agent. Each critic gets the concatenated
        # observations/actions of each agent of the same type as the agent.
        # TODO (dries): Add the option of directly getting state information
        #  from the environment.
        # TODO (dries): Make the critic more general and allow the critic network to get
        #  observations/actions inputs of agents from different types as well.
        #  Maybe use a multiplexer to do so?
        for agent_type, agents in agents_by_type.items():
            critic_obs_shape = list(copy.copy(self._embed_specs[agent_type].shape))
            critic_obs_shape.insert(0, len(agents))
            critic_act_shape = list(
                copy.copy(self._agent_specs[agents[0]].actions.shape)
            )
            critic_act_shape.insert(0, len(agents))
            obs_specs_per_type[agent_type] = tf.TensorSpec(
                shape=critic_obs_shape,
                dtype=tf.dtypes.float32,
            )
            action_specs_per_type[agent_type] = tf.TensorSpec(
                shape=critic_act_shape,
                dtype=tf.dtypes.float32,
            )

        critic_obs_specs = {}
        critic_act_specs = {}
        for agent_key in self._critic_agent_keys:
            agent_type = agent_key.split("_")[0]
            # Get observation and action spec for critic.
            critic_obs_specs[agent_key] = obs_specs_per_type[agent_type]
            critic_act_specs[agent_key] = action_specs_per_type[agent_type]
        return critic_obs_specs, critic_act_specs
