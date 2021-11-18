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
from mava.components.tf.architectures.decentralised import (
    DecentralisedPolicyActor,
    DecentralisedQValueActorCritic,
)


class StateBasedPolicyActor(DecentralisedPolicyActor):
    """Multi-agent actor architecture using
    environment state information."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
    ):
        super().__init__(
            environment_spec=environment_spec,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            agent_net_keys=agent_net_keys,
        )

    def _get_actor_specs(
        self,
    ) -> Dict[str, acme_specs.Array]:
        obs_specs_per_type: Dict[str, acme_specs.Array] = {}

        agents_by_type = self._env_spec.get_agents_by_type()

        for agent_type, agents in agents_by_type.items():
            actor_state_shape = self._env_spec.get_extra_specs()["s_t"].shape
            obs_specs_per_type[agent_type] = tf.TensorSpec(
                shape=actor_state_shape,
                dtype=tf.dtypes.float32,
            )

        actor_obs_specs = {}
        for agent_key in self._agents:
            agent_net_key = self._agent_net_keys[agent_key]
            # Get observation spec for actor.
            actor_obs_specs[agent_key] = obs_specs_per_type[agent_net_key]
        return actor_obs_specs


class StateBasedQValueCritic(DecentralisedQValueActorCritic):
    """Multi-agent actor critic architecture with a critic using
    environment state information."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
    ):
        super().__init__(
            environment_spec=environment_spec,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            agent_net_keys=agent_net_keys,
        )

    def _get_critic_specs(
        self,
    ) -> Tuple[Dict[str, acme_specs.Array], Dict[str, acme_specs.Array]]:
        action_specs_per_type: Dict[str, acme_specs.Array] = {}

        agents_by_type = self._env_spec.get_agents_by_type()

        # Create one critic per agent. Each critic gets
        # absolute state information of the environment.
        critic_state_shape = self._env_spec.get_extra_specs()["s_t"].shape
        critic_obs_spec = tf.TensorSpec(
            shape=critic_state_shape,
            dtype=tf.dtypes.float32,
        )
        for agent_type, agents in agents_by_type.items():
            critic_act_shape = list(
                copy.copy(self._agent_specs[agents[0]].actions.shape)
            )
            critic_act_shape.insert(0, len(agents))
            action_specs_per_type[agent_type] = tf.TensorSpec(
                shape=critic_act_shape,
                dtype=tf.dtypes.float32,
            )

        critic_obs_specs = {}
        critic_act_specs = {}
        for agent_key in self._agents:
            agent_type = agent_key.split("_")[0]
            # Get observation and action spec for critic.
            critic_obs_specs[agent_key] = critic_obs_spec
            critic_act_specs[agent_key] = action_specs_per_type[agent_type]
        return critic_obs_specs, critic_act_specs


class StateBasedQValueActorCritic(  # type: ignore
    StateBasedPolicyActor, StateBasedQValueCritic
):
    """Multi-agent actor critic architecture where both actor policies
    and critics use environment state information"""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
    ):

        StateBasedQValueCritic.__init__(
            self,
            environment_spec=environment_spec,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            agent_net_keys=agent_net_keys,
        )


class StateBasedQValueSingleActionCritic(DecentralisedQValueActorCritic):
    """Multi-agent actor critic architecture with a critic using
    environment state information. For this state-based critic
    only one action gets fed in. This allows the critic
    to only focus on the one agent's reward function, but
    requires that the state information is
    egocentric."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
    ):
        super().__init__(
            environment_spec=environment_spec,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            agent_net_keys=agent_net_keys,
        )

    def _get_critic_specs(
        self,
    ) -> Tuple[Dict[str, acme_specs.Array], Dict[str, acme_specs.Array]]:
        action_specs_per_type: Dict[str, acme_specs.Array] = {}

        agents_by_type = self._env_spec.get_agents_by_type()

        # Create one critic per agent. Each critic gets
        # absolute state information of the environment.
        critic_env_state_spec = self._env_spec.get_extra_specs()["env_states"]
        if type(critic_env_state_spec) == dict:
            critic_env_state_spec = list(critic_env_state_spec.values())[0]

        if type(critic_env_state_spec) != list:
            critic_env_state_spec = [critic_env_state_spec]

        critic_obs_spec = []
        for spec in critic_env_state_spec:
            critic_obs_spec.append(
                tf.TensorSpec(
                    shape=spec.shape,
                    dtype=tf.dtypes.float32,
                )
            )

        for agent_type, agents in agents_by_type.items():
            # Only feed in the main agent's policy action.
            critic_act_shape = list(
                copy.copy(self._agent_specs[agents[0]].actions.shape)
            )
            action_specs_per_type[agent_type] = tf.TensorSpec(
                shape=critic_act_shape,
                dtype=tf.dtypes.float32,
            )

        critic_obs_specs = {}
        critic_act_specs = {}
        for agent_key in self._agents:
            agent_type = agent_key.split("_")[0]
            # Get observation and action spec for critic.
            critic_obs_specs[agent_key] = critic_obs_spec
            critic_act_specs[agent_key] = action_specs_per_type[agent_type]
        return critic_obs_specs, critic_act_specs
