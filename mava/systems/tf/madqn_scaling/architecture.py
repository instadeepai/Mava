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
from typing import Dict

import sonnet as snt
from acme.tf import utils as tf2_utils

from mava import specs as mava_specs
from mava.components.tf.architectures import (
    BaseArchitecture,
)
from mava.types import OLT

class DecentralisedValueActor(BaseArchitecture):
    """Decentralised (independent) value-based multi-agent actor architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        value_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
        net_spec_keys: Dict[str, str] = {},

    ):
        self._env_spec = environment_spec
        self._agents = self._env_spec.get_agent_ids()
        self._agent_types = self._env_spec.get_agent_types()
        self._agent_specs = self._env_spec.get_agent_specs()
        self._agent_type_specs = self._env_spec.get_agent_type_specs()

        self._observation_networks = observation_networks
        self._value_networks = value_networks
        self._agent_net_keys = agent_net_keys
        self._n_agents = len(self._agents)

        if not net_spec_keys:
            # Check if the agents use all the networks.
            assert len(self._value_networks.keys()) == len(
                set(self._agent_net_keys.values())
            )
            net_spec_keys = {
                self._agent_net_keys[agent]: agent for agent in self._agents
            }
        self._net_spec_keys = net_spec_keys

        self._create_target_networks()

    def _create_target_networks(self) -> None:
        # Create target networks
        self._target_value_networks = copy.deepcopy(self._value_networks)
        self._target_observation_networks = copy.deepcopy(self._observation_networks)

    def _get_actor_specs(self) -> Dict[str, OLT]:
        actor_obs_specs = {}
        for agent_key in self._agents:
            # Get observation spec for policy.
            actor_obs_specs[agent_key] = self._agent_specs[
                agent_key
            ].observations.observation
        return actor_obs_specs

    def create_actor_variables(self) -> Dict[str, Dict[str, snt.Module]]:

        actor_networks: Dict[str, Dict[str, snt.Module]] = {
            "values": {},
            "target_values": {},
            "observations": {},
            "target_observations": {}
        }

        # get actor specs
        actor_obs_specs = self._get_actor_specs()

        # create policy variables for each agent
        for agent_key in self._agents:
            agent_net_key = self._agent_net_keys[agent_key]
            obs_spec = actor_obs_specs[agent_key]
            # Create variables for obs and value networks.
            embed_spec = tf2_utils.create_variables(self._observation_networks[agent_net_key], [obs_spec])
            tf2_utils.create_variables(self._value_networks[agent_net_key], [embed_spec])

            # Create target network variables
            embed_spec = tf2_utils.create_variables(
                self._target_observation_networks[agent_net_key], [obs_spec]
            )
            tf2_utils.create_variables(self._target_value_networks[agent_net_key], [embed_spec])

        actor_networks["values"] = self._value_networks
        actor_networks["target_values"] = self._target_value_networks
        actor_networks["observations"] = self._observation_networks
        actor_networks["target_observations"] = self._target_observation_networks

        return actor_networks

    def create_system(
        self,
    ) -> Dict[str, Dict[str, snt.Module]]:
        networks = self.create_actor_variables()
        return networks

    # TODO (Claude) Add action selector maybe? 
    def create_behaviour_policy(self) -> Dict[str, snt.Module]:
        behaviour_policy_networks: Dict[str, snt.Module] = {}
        for net_key in self._value_networks.keys():
            snt_module = type(self._value_networks[net_key])
            behaviour_policy_networks[net_key] = snt_module(
                [
                    self._observation_networks[net_key],
                    self._value_networks[net_key],
                ]
            )
        return behaviour_policy_networks