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

"""Decentralised architectures for multi-agent RL systems"""

import copy
from typing import Dict, List, Tuple

import sonnet as snt
from acme import specs
from acme.tf import utils as tf2_utils

from mava.components.tf.architectures import BaseArchitecture


class DecentralisedActorCritic(BaseArchitecture):
    """Decentralised (independent) multi-agent actor critic architecture."""

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        environment_spec: specs.EnvironmentSpec,
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
    ):
        self._agents = agents
        self._agent_types = agent_types
        self._environment_spec = environment_spec
        self._policy_networks = policy_networks
        self._critic_networks = critic_networks
        self._observation_networks = observation_networks
        self._shared_weights = shared_weights
        self._agent_keys = self._agent_types if self._shared_weights else self._agents
        self._n_agents = len(self._agents)

    def _create_target_networks(
        self, agent_key: str
    ) -> Tuple[snt.Module, snt.Module, snt.Module]:

        # Create target networks.
        target_policy_network = copy.deepcopy(self._policy_networks[agent_key])
        target_critic_network = copy.deepcopy(self._critic_networks[agent_key])
        target_observation_network = copy.deepcopy(
            self._observation_networks[agent_key]
        )
        return target_policy_network, target_critic_network, target_observation_network

    def _get_specs(
        self, agent_key: str
    ) -> Tuple[specs.Array, specs.Array, specs.Array, Tuple[specs.Array, specs.Array]]:

        # Get observation and action specs.
        act_spec = self._environment_spec[agent_key].actions
        obs_spec = self._environment_spec[agent_key].observations
        emb_spec = tf2_utils.create_variables(
            self._observation_networks[agent_key], [obs_spec]
        )
        crit_spec = (obs_spec, act_spec)
        return act_spec, obs_spec, emb_spec, crit_spec

    def create_system(self) -> Dict[str, Dict[str, snt.Module]]:
        networks: Dict[str, Dict[str, snt.Module]] = {
            "policies": {},
            "critics": {},
            "observations": {},
            "target_policies": {},
            "target_critics": {},
            "target_observations": {},
        }
        for agent_key in self._agent_keys:

            # get specs
            act_spec, obs_spec, emb_spec, crit_spec = self._get_specs(agent_key)
            # create target networks
            (
                target_policy_network,
                target_critic_network,
                target_observation_network,
            ) = self._create_target_networks(agent_key)

            # critic specs
            crit_obs_spec, crit_act_spec = crit_spec

            # Create variables.
            tf2_utils.create_variables(self._policy_networks[agent_key], [emb_spec])
            # TODO Remove [0] - this is a temp fix to get this running
            tf2_utils.create_variables(
                self._critic_networks[agent_key], [crit_obs_spec[0], crit_act_spec[0]]
            )

            # create target network variables
            tf2_utils.create_variables(target_policy_network, [emb_spec])
            tf2_utils.create_variables(target_observation_network, [obs_spec])
            # TODO Remove [0] - this is a temp fix to get this running
            tf2_utils.create_variables(
                target_critic_network, [crit_obs_spec[0], crit_act_spec[0]]
            )
            networks["policies"][agent_key] = self._policy_networks[agent_key]
            networks["critics"][agent_key] = self._critic_networks[agent_key]
            networks["observations"][agent_key] = self._observation_networks[agent_key]
            networks["target_policies"][agent_key] = target_policy_network
            networks["target_critics"][agent_key] = target_critic_network
            networks["target_observations"][agent_key] = target_observation_network
        return networks
