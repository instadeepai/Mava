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

from typing import Dict, Tuple

import sonnet as snt
from acme import specs

from mava.components.tf.architectures.decentralised import DecentralisedActorCritic


class CentralisedActorCritic(DecentralisedActorCritic):
    """Centralised multi-agent actor critic architecture."""

    def __init__(
        self,
        environment_spec: specs.MAEnvironmentSpec,
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
    ):
        super().__init__(
            environment_spec=environment_spec,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            observation_networks=observation_networks,
            shared_weights=shared_weights,
        )

    def _get_centralised_spec(self) -> Dict[str, specs.Array]:
        specs_per_type: Dict[str, specs.Array] = {}
        agents_by_type = self._env_spec.get_agents_by_type()
        for agent_type, agents in agents_by_type.items():
            critic_spec = self._agent_specs[agents[0]]
            critic_obs_shape = [
                0 for dim in self._agent_specs[agents[0]].observations.shape
            ]
            critic_act_shape = [0 for dim in self._agent_specs[agents[0]].actions.shape]
            for agent in agents:
                for obs_dim in range(len(critic_obs_shape)):
                    critic_obs_shape[obs_dim] += self._agent_specs[
                        agent
                    ].observations.shape[obs_dim]
                for act_dim in range(len(critic_act_shape)):
                    critic_act_shape[act_dim] += self._agent_specs[agent].actions.shape[
                        act_dim
                    ]
            critic_spec.observations._shape = tuple(critic_obs_shape)
            critic_spec.actions._shape = tuple(critic_act_shape)
            for agent in agents:
                specs_per_type[agent] = critic_spec
        return specs_per_type

    def _get_critic_specs(
        self,
    ) -> Tuple[Dict[str, specs.Array], Dict[str, specs.Array]]:
        centralised_specs = self._get_centralised_spec()

        critic_obs_specs = {}
        critic_act_specs = {}

        for agent_key in self._critic_agent_keys:
            agent_spec_key = f"{agent_key}_0" if self._shared_weights else agent_key

            # Get observation and action spec for critic.
            critic_obs_specs[agent_key] = centralised_specs[agent_spec_key].observations
            critic_act_specs[agent_key] = centralised_specs[agent_spec_key].actions
        return critic_obs_specs, critic_act_specs
