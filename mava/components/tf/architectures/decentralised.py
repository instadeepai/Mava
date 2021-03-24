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
from typing import Dict, Tuple

import sonnet as snt
from acme import specs as acme_specs
from acme.tf import utils as tf2_utils

from mava import specs
from mava.components.tf.architectures import BaseActorCritic


class DecentralisedActorCritic(BaseActorCritic):
    """Decentralised (independent) multi-agent actor critic architecture."""

    def __init__(
        self,
        environment_spec: specs.MAEnvironmentSpec,
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
    ):
        self._env_spec = environment_spec
        self._agents = self._env_spec.get_agent_ids()
        self._agent_types = self._env_spec.get_agent_types()
        self._agent_specs = self._env_spec.get_agent_specs()
        self._agent_type_specs = self._env_spec.get_agent_type_specs()

        self._policy_networks = policy_networks
        self._critic_networks = critic_networks
        self._observation_networks = observation_networks
        self._shared_weights = shared_weights
        self._actor_agent_keys = (
            self._agent_types if self._shared_weights else self._agents
        )
        self._critic_agent_keys = self._actor_agent_keys
        self._n_agents = len(self._agents)

        self._create_target_networks()

    def _create_target_networks(self) -> None:
        # create target behaviour networks
        self._target_policy_networks = copy.deepcopy(self._policy_networks)
        self._target_observation_networks = copy.deepcopy(self._observation_networks)

        # create target critic networks
        self._target_critic_networks = copy.deepcopy(self._critic_networks)

    def _get_actor_specs(self) -> Dict[str, acme_specs.Array]:
        actor_obs_specs = {}
        for agent_key in self._actor_agent_keys:
            agent_spec_key = f"{agent_key}_0" if self._shared_weights else agent_key

            # Get observation spec for policy.
            actor_obs_specs[agent_key] = self._agent_specs[agent_spec_key].observations
        return actor_obs_specs

    def _get_critic_specs(
        self,
    ) -> Tuple[Dict[str, acme_specs.Array], Dict[str, acme_specs.Array]]:
        critic_obs_specs = {}
        critic_act_specs = {}
        for agent_key in self._critic_agent_keys:
            agent_spec_key = f"{agent_key}_0" if self._shared_weights else agent_key

            # Get observation and action spec for critic.
            critic_obs_specs[agent_key] = self._agent_specs[agent_spec_key].observations
            critic_act_specs[agent_key] = self._agent_specs[agent_spec_key].actions
        return critic_obs_specs, critic_act_specs

    def create_actor_variables(self) -> Dict[str, Dict[str, snt.Module]]:

        actor_networks: Dict[str, Dict[str, snt.Module]] = {
            "policies": {},
            "observations": {},
            "target_policies": {},
            "target_observations": {},
        }

        # get actor specs
        actor_obs_specs = self._get_actor_specs()

        # create policy variables for each agent
        for agent_key in self._actor_agent_keys:

            obs_spec = actor_obs_specs[agent_key]
            emb_spec = tf2_utils.create_variables(
                self._observation_networks[agent_key], [obs_spec]
            )

            # Create variables.
            tf2_utils.create_variables(self._policy_networks[agent_key], [emb_spec])

            # create target network variables
            tf2_utils.create_variables(
                self._target_policy_networks[agent_key], [emb_spec]
            )
            tf2_utils.create_variables(
                self._target_observation_networks[agent_key], [obs_spec]
            )

        actor_networks["policies"] = self._policy_networks
        actor_networks["observations"] = self._observation_networks
        actor_networks["target_policies"] = self._target_policy_networks
        actor_networks["target_observations"] = self._target_observation_networks

        return actor_networks

    def create_critic_variables(self) -> Dict[str, Dict[str, snt.Module]]:

        critic_networks: Dict[str, Dict[str, snt.Module]] = {
            "critics": {},
            "target_critics": {},
        }

        # get critic specs
        obs_specs, act_specs = self._get_critic_specs()

        # create critics
        for agent_key in self._critic_agent_keys:

            # get specs
            obs_spec = obs_specs[agent_key]
            act_spec = act_specs[agent_key]

            # Create variables.
            tf2_utils.create_variables(
                self._critic_networks[agent_key], [obs_spec, act_spec]
            )

            # create target network variables
            tf2_utils.create_variables(
                self._target_critic_networks[agent_key], [obs_spec, act_spec]
            )

        critic_networks["critics"][agent_key] = self._critic_networks
        critic_networks["target_critics"][agent_key] = self._target_critic_networks

        return critic_networks

    def create_system(
        self,
    ) -> Dict[str, Dict[str, snt.Module]]:
        networks = self.create_actor_variables()
        critic_networks = self.create_critic_variables()
        networks.update(critic_networks)
        return networks
