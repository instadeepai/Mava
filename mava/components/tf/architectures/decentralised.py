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
from typing import Any, Dict, Tuple

import sonnet as snt
from acme import specs as acme_specs
from acme.tf import utils as tf2_utils

from mava import specs as mava_specs
from mava.components.tf.architectures import (
    BaseActorCritic,
    BaseArchitecture,
    BasePolicyArchitecture,
)
from mava.types import OLT


class DecentralisedValueActor(BaseArchitecture):
    """Decentralised (independent) value-based multi-agent actor architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        value_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
    ):
        self._env_spec = environment_spec
        self._agents = self._env_spec.get_agent_ids()
        self._agent_types = self._env_spec.get_agent_types()
        self._agent_specs = self._env_spec.get_agent_specs()
        self._agent_type_specs = self._env_spec.get_agent_type_specs()

        self._value_networks = value_networks
        self._agent_net_keys = agent_net_keys
        self._n_agents = len(self._agents)

        self._create_target_networks()

    def _create_target_networks(self) -> None:
        # create target behaviour networks
        self._target_value_networks = copy.deepcopy(self._value_networks)

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
        }

        # get actor specs
        actor_obs_specs = self._get_actor_specs()

        # create policy variables for each agent
        for agent_key in self._agents:
            agent_net_key = self._agent_net_keys[agent_key]
            obs_spec = actor_obs_specs[agent_key]
            # Create variables for value and policy networks.
            tf2_utils.create_variables(self._value_networks[agent_net_key], [obs_spec])

            # create target value network variables
            tf2_utils.create_variables(
                self._target_value_networks[agent_net_key], [obs_spec]
            )

        actor_networks["values"] = self._value_networks
        actor_networks["target_values"] = self._target_value_networks

        return actor_networks

    def create_system(
        self,
    ) -> Dict[str, Dict[str, snt.Module]]:
        networks = self.create_actor_variables()
        return networks


class DecentralisedPolicyActor(BasePolicyArchitecture):
    """Decentralised (independent) policy gradient multi-agent actor architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
    ):
        self._env_spec = environment_spec
        self._agents = self._env_spec.get_agent_ids()
        self._agent_types = self._env_spec.get_agent_types()
        self._agent_specs = self._env_spec.get_agent_specs()
        self._agent_type_specs = self._env_spec.get_agent_type_specs()

        self._observation_networks = observation_networks
        self._policy_networks = policy_networks
        self._agent_net_keys = agent_net_keys
        self._n_agents = len(self._agents)
        self._embed_specs: Dict[str, Any] = {}

        self._create_target_networks()

    def _create_target_networks(self) -> None:
        # create target behaviour networks
        self._target_policy_networks = copy.deepcopy(self._policy_networks)
        self._target_observation_networks = copy.deepcopy(self._observation_networks)

    def _get_actor_specs(self) -> Dict[str, acme_specs.Array]:
        actor_obs_specs = {}
        for agent_key in self._agents:
            # Get observation spec for policy.
            actor_obs_specs[agent_key] = self._agent_specs[
                agent_key
            ].observations.observation
        return actor_obs_specs

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
        for agent_key in self._agents:
            agent_net_key = self._agent_net_keys[agent_key]

            obs_spec = actor_obs_specs[agent_key]
            emb_spec = tf2_utils.create_variables(
                self._observation_networks[agent_net_key], [obs_spec]
            )
            self._embed_specs[agent_key] = emb_spec

            # Create variables.
            tf2_utils.create_variables(self._policy_networks[agent_net_key], [emb_spec])

            # create target network variables
            tf2_utils.create_variables(
                self._target_policy_networks[agent_net_key], [emb_spec]
            )
            tf2_utils.create_variables(
                self._target_observation_networks[agent_net_key], [obs_spec]
            )

        actor_networks["policies"] = self._policy_networks
        actor_networks["observations"] = self._observation_networks
        actor_networks["target_policies"] = self._target_policy_networks
        actor_networks["target_observations"] = self._target_observation_networks

        return actor_networks

    def create_behaviour_policy(self) -> Dict[str, snt.Module]:
        behaviour_policy_networks: Dict[str, snt.Module] = {}
        for net_key in self._policy_networks.keys():
            snt_module = type(self._policy_networks[net_key])
            behaviour_policy_networks[net_key] = snt_module(
                [
                    self._observation_networks[net_key],
                    self._policy_networks[net_key],
                ]
            )
        return behaviour_policy_networks

    def create_system(
        self,
    ) -> Dict[str, Dict[str, snt.Module]]:
        networks = self.create_actor_variables()
        return networks


class DecentralisedValueActorCritic(BaseActorCritic):
    """Decentralised (independent) multi-agent actor critic architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
        net_spec_keys: Dict[str, str] = {},
    ):
        self._env_spec = environment_spec
        self._agents = self._env_spec.get_agent_ids()  # All agend ids
        self._agent_specs = (
            self._env_spec.get_agent_specs()
        )  # Each agent's environment interaction specification

        self._observation_networks = observation_networks
        self._policy_networks = policy_networks
        self._critic_networks = critic_networks
        self._agent_net_keys = agent_net_keys
        self._net_keys = self._policy_networks.keys()
        self._n_agents = len(self._agents)
        self._embed_specs: Dict[str, Any] = {}

        if not net_spec_keys:
            # Check if the agents use all the networks.
            assert len(self._policy_networks.keys()) == len(
                set(self._agent_net_keys.values())
            )
            net_spec_keys = {
                self._agent_net_keys[agent]: agent for agent in self._agents
            }
        self._net_spec_keys = net_spec_keys

        self._create_target_networks()

    def _create_target_networks(self) -> None:
        # create target behaviour networks
        self._target_policy_networks = copy.deepcopy(self._policy_networks)
        self._target_observation_networks = copy.deepcopy(self._observation_networks)

        # create target critic networks
        self._target_critic_networks = copy.deepcopy(self._critic_networks)

    def _get_actor_specs(self) -> Dict[str, acme_specs.Array]:
        actor_obs_specs = {}
        for agent_key in self._agents:
            # Get observation spec for policy.
            actor_obs_specs[agent_key] = self._agent_specs[
                agent_key
            ].observations.observation
        return actor_obs_specs

    def _get_critic_specs(
        self,
    ) -> Tuple[Dict[str, acme_specs.Array], Dict[str, acme_specs.Array]]:
        return self._embed_specs, {}

    def create_actor_variables(self) -> Dict[str, Dict[str, snt.Module]]:
        # get actor specs
        actor_obs_specs = self._get_actor_specs()

        # create policy variables for each agent
        for net_key in self._net_keys:
            agent_key = self._net_spec_keys[net_key]
            obs_spec = actor_obs_specs[agent_key]
            emb_spec = tf2_utils.create_variables(
                self._observation_networks[net_key], [obs_spec]
            )
            self._embed_specs[agent_key] = emb_spec

            # Create variables.
            tf2_utils.create_variables(self._policy_networks[net_key], [emb_spec])

            # create target network variables
            tf2_utils.create_variables(
                self._target_policy_networks[net_key], [emb_spec]
            )
            tf2_utils.create_variables(
                self._target_observation_networks[net_key], [obs_spec]
            )

        actor_networks: Dict[str, Dict[str, snt.Module]] = {
            "policies": self._policy_networks,
            "observations": self._observation_networks,
            "target_policies": self._target_policy_networks,
            "target_observations": self._target_observation_networks,
        }

        return actor_networks

    def create_critic_variables(self) -> Dict[str, Dict[str, snt.Module]]:
        # get critic specs
        embed_specs, _ = self._get_critic_specs()

        # create critics
        for net_key in self._net_keys:
            agent_key = self._net_spec_keys[net_key]

            # get specs
            emb_spec = embed_specs[agent_key]

            # Create variables.
            tf2_utils.create_variables(self._critic_networks[net_key], [emb_spec])

            # create target network variables
            tf2_utils.create_variables(
                self._target_critic_networks[net_key], [emb_spec]
            )

        critic_networks: Dict[str, Dict[str, snt.Module]] = {
            "critics": self._critic_networks,
            "target_critics": self._target_critic_networks,
        }
        return critic_networks

    def create_behaviour_policy(self) -> Dict[str, snt.Module]:
        behaviour_policy_networks: Dict[str, snt.Module] = {}
        for net_key in self._net_keys:
            snt_module = type(self._policy_networks[net_key])
            behaviour_policy_networks[net_key] = snt_module(
                [
                    self._observation_networks[net_key],
                    self._policy_networks[net_key],
                ]
            )
        return behaviour_policy_networks

    def create_system(
        self,
    ) -> Dict[str, Dict[str, snt.Module]]:
        networks = self.create_actor_variables()
        critic_networks = self.create_critic_variables()
        networks.update(critic_networks)
        return networks


class DecentralisedQValueActorCritic(DecentralisedValueActorCritic):
    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
        net_spec_keys: Dict[str, str] = {},
    ):
        super().__init__(
            environment_spec=environment_spec,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            agent_net_keys=agent_net_keys,
            net_spec_keys=net_spec_keys,
        )

    def _get_critic_specs(
        self,
    ) -> Tuple[Dict[str, acme_specs.Array], Dict[str, acme_specs.Array]]:
        critic_act_specs = {}
        for agent_key in self._agents:
            # Get observation and action spec for critic.
            critic_act_specs[agent_key] = self._agent_specs[agent_key].actions
        return self._embed_specs, critic_act_specs

    def create_critic_variables(self) -> Dict[str, Dict[str, snt.Module]]:
        # get critic specs
        embed_specs, act_specs = self._get_critic_specs()

        # create critics
        for net_key in self._net_keys:
            agent_key = self._net_spec_keys[net_key]

            # get specs
            emb_spec = embed_specs[agent_key]
            act_spec = act_specs[agent_key]

            # Create variables.
            tf2_utils.create_variables(
                self._critic_networks[net_key], [emb_spec, act_spec]
            )

            # create target network variables
            tf2_utils.create_variables(
                self._target_critic_networks[net_key], [emb_spec, act_spec]
            )

        critic_networks: Dict[str, Dict[str, snt.Module]] = {
            "critics": self._critic_networks,
            "target_critics": self._target_critic_networks,
        }
        return critic_networks
