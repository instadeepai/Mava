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

from typing import Dict, List, Tuple

import numpy as np
import sonnet as snt
from acme import specs
from acme.tf import utils as tf2_utils

from mava.components.tf.architectures.decentralised import DecentralisedActorCritic


class CentralisedActorCritic(DecentralisedActorCritic):
    """Centralised multi-agent actor critic architecture."""

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
        super().__init__(
            agents=agents,
            agent_types=agent_types,
            environment_spec=environment_spec,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            observation_networks=observation_networks,
            shared_weights=shared_weights,
        )

    def _get_specs(
        self, agent_key: str
    ) -> Tuple[specs.Array, specs.Array, specs.Array, Tuple[specs.Array, specs.Array]]:

        # Get observation and action specs.
        act_spec = self._environment_spec[agent_key].actions
        obs_spec = self._environment_spec[agent_key].observations
        emb_spec = tf2_utils.create_variables(
            self._observation_networks[agent_key], [obs_spec]
        )

        # create centralised critic spec
        crit_obs_spec = np.tile(emb_spec, self._n_agents)
        crit_act_spec = np.tile(act_spec, self._n_agents)
        crit_spec = (crit_obs_spec, crit_act_spec)
        return act_spec, obs_spec, emb_spec, crit_spec
