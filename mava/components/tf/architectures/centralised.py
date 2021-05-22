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
    DecentralisedSoftQValueActorCritic,
    DecentralisedValueActorCritic,
)


class CentralisedPolicyActor(DecentralisedPolicyActor):
    """Centralised multi-agent actor architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        shared_weights: bool = True,
    ):
        super().__init__(
            environment_spec=environment_spec,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            shared_weights=shared_weights,
        )

    def _get_actor_specs(
        self,
    ) -> Dict[str, acme_specs.Array]:
        obs_specs_per_type: Dict[str, acme_specs.Array] = {}

        agents_by_type = self._env_spec.get_agents_by_type()

        for agent_type, agents in agents_by_type.items():
            actor_obs_shape = list(
                copy.copy(
                    self._agent_type_specs[agent_type].observations.observation.shape
                )
            )
            actor_obs_shape.insert(0, len(agents))
            obs_specs_per_type[agent_type] = tf.TensorSpec(
                shape=actor_obs_shape,
                dtype=tf.dtypes.float32,
            )

        actor_obs_specs = {}
        for agent_key in self._actor_agent_keys:
            agent_type = agent_key.split("_")[0]
            # Get observation spec for actor.
            actor_obs_specs[agent_key] = obs_specs_per_type[agent_type]
        return actor_obs_specs


class CentralisedValueCritic(DecentralisedValueActorCritic):
    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
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

    def _get_critic_specs(
        self,
    ) -> Tuple[Dict[str, acme_specs.Array], Dict[str, acme_specs.Array]]:
        obs_specs_per_type: Dict[str, acme_specs.Array] = {}
        agents_by_type = self._env_spec.get_agents_by_type()

        for agent_type, agents in agents_by_type.items():
            agent_key = agent_type if self._shared_weights else agents[0]
            critic_obs_shape = list(copy.copy(self._embed_specs[agent_key].shape))
            critic_obs_shape.insert(0, len(agents))
            obs_specs_per_type[agent_type] = tf.TensorSpec(
                shape=critic_obs_shape,
                dtype=tf.dtypes.float32,
            )

        critic_obs_specs = {}
        for agent_key in self._critic_agent_keys:
            agent_type = agent_key.split("_")[0]
            # Get observation and action spec for critic.
            critic_obs_specs[agent_key] = obs_specs_per_type[agent_type]
        return critic_obs_specs, {}


class CentralisedQValueCritic(DecentralisedQValueActorCritic):
    """Centralised multi-agent actor critic architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
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

    def _get_critic_specs(
        self,
    ) -> Tuple[Dict[str, acme_specs.Array], Dict[str, acme_specs.Array]]:
        obs_specs_per_type: Dict[str, acme_specs.Array] = {}
        action_specs_per_type: Dict[str, acme_specs.Array] = {}

        agents_by_type = self._env_spec.get_agents_by_type()

        for agent_type, agents in agents_by_type.items():
            agent_key = agent_type if self._shared_weights else agents[0]
            critic_obs_shape = list(copy.copy(self._embed_specs[agent_key].shape))
            critic_obs_shape.insert(0, len(agents))
            obs_specs_per_type[agent_type] = tf.TensorSpec(
                shape=critic_obs_shape,
                dtype=tf.dtypes.float32,
            )

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
        for agent_key in self._critic_agent_keys:
            agent_type = agent_key.split("_")[0]
            # Get observation and action spec for critic.
            critic_obs_specs[agent_key] = obs_specs_per_type[agent_type]
            critic_act_specs[agent_key] = action_specs_per_type[agent_type]
        return critic_obs_specs, critic_act_specs


class CentralisedSoftQValueCritic(DecentralisedSoftQValueActorCritic):
    """Centralised multi-agent soft actor critic architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_V_networks: Dict[str, snt.Module],
        critic_Q_1_networks: Dict[str, snt.Module],
        critic_Q_2_networks: Dict[str, snt.Module],
        shared_weights: bool = True,
    ):
        super().__init__(
            environment_spec=environment_spec,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            critic_V_networks=critic_V_networks,
            critic_Q_1_networks=critic_Q_1_networks,
            critic_Q_2_networks=critic_Q_2_networks,
            shared_weights=shared_weights,
        )

    def _get_critic_V_specs(
        self,
    ) -> Tuple[Dict[str, acme_specs.Array], Dict[str, acme_specs.Array]]:
        obs_specs_per_type: Dict[str, acme_specs.Array] = {}
        action_specs_per_type: Dict[str, acme_specs.Array] = {}

        agents_by_type = self._env_spec.get_agents_by_type()

        for agent_type, agents in agents_by_type.items():
            critic_V_obs_shape = list(copy.copy(self._embed_specs[agent_type].shape))
            critic_V_obs_shape.insert(0, len(agents))
            obs_specs_per_type[agent_type] = tf.TensorSpec(
                shape=critic_V_obs_shape,
                dtype=tf.dtypes.float32,
            )

            critic_V_act_shape = list(
                copy.copy(self._agent_specs[agents[0]].actions.shape)
            )
            critic_V_act_shape.insert(0, len(agents))
            action_specs_per_type[agent_type] = tf.TensorSpec(
                shape=critic_V_act_shape,
                dtype=tf.dtypes.float32,
            )

        critic_V_obs_specs = {}
        critic_V_act_specs = {}
        for agent_key in self._critic_V_agent_keys:
            agent_type = agent_key.split("_")[0]
            # Get observation and action spec for critic_V.
            critic_V_obs_specs[agent_key] = obs_specs_per_type[agent_type]
            critic_V_act_specs[agent_key] = action_specs_per_type[agent_type]
        return critic_V_obs_specs, critic_V_act_specs

    def _get_critic_Q_1_specs(
        self,
    ) -> Tuple[Dict[str, acme_specs.Array], Dict[str, acme_specs.Array]]:
        obs_specs_per_type: Dict[str, acme_specs.Array] = {}
        action_specs_per_type: Dict[str, acme_specs.Array] = {}

        agents_by_type = self._env_spec.get_agents_by_type()

        for agent_type, agents in agents_by_type.items():
            critic_Q_1_obs_shape = list(copy.copy(self._embed_specs[agent_type].shape))
            critic_Q_1_obs_shape.insert(0, len(agents))
            obs_specs_per_type[agent_type] = tf.TensorSpec(
                shape=critic_Q_1_obs_shape,
                dtype=tf.dtypes.float32,
            )

            critic_Q_1_act_shape = list(
                copy.copy(self._agent_specs[agents[0]].actions.shape)
            )
            critic_Q_1_act_shape.insert(0, len(agents))
            action_specs_per_type[agent_type] = tf.TensorSpec(
                shape=critic_Q_1_act_shape,
                dtype=tf.dtypes.float32,
            )

        critic_Q_1_obs_specs = {}
        critic_Q_1_act_specs = {}
        for agent_key in self._critic_Q_1_agent_keys:
            agent_type = agent_key.split("_")[0]
            # Get observation and action spec for critic_Q_1.
            critic_Q_1_obs_specs[agent_key] = obs_specs_per_type[agent_type]
            critic_Q_1_act_specs[agent_key] = action_specs_per_type[agent_type]
        return critic_Q_1_obs_specs, critic_Q_1_act_specs

    def _get_critic_Q_2_specs(
        self,
    ) -> Tuple[Dict[str, acme_specs.Array], Dict[str, acme_specs.Array]]:
        obs_specs_per_type: Dict[str, acme_specs.Array] = {}
        action_specs_per_type: Dict[str, acme_specs.Array] = {}

        agents_by_type = self._env_spec.get_agents_by_type()

        for agent_type, agents in agents_by_type.items():
            critic_Q_2_obs_shape = list(copy.copy(self._embed_specs[agent_type].shape))
            critic_Q_2_obs_shape.insert(0, len(agents))
            obs_specs_per_type[agent_type] = tf.TensorSpec(
                shape=critic_Q_2_obs_shape,
                dtype=tf.dtypes.float32,
            )

            critic_Q_2_act_shape = list(
                copy.copy(self._agent_specs[agents[0]].actions.shape)
            )
            critic_Q_2_act_shape.insert(0, len(agents))
            action_specs_per_type[agent_type] = tf.TensorSpec(
                shape=critic_Q_2_act_shape,
                dtype=tf.dtypes.float32,
            )

        critic_Q_2_obs_specs = {}
        critic_Q_2_act_specs = {}
        for agent_key in self._critic_Q_2_agent_keys:
            agent_type = agent_key.split("_")[0]
            # Get observation and action spec for critic_Q_2.
            critic_Q_2_obs_specs[agent_key] = obs_specs_per_type[agent_type]
            critic_Q_2_act_specs[agent_key] = action_specs_per_type[agent_type]
        return critic_Q_2_obs_specs, critic_Q_2_act_specs


# TODO (Arnu): remove mypy type ignore once we can handle type checking for
# nested/multiple inheritance
class CentralisedQValueActorCritic(  # type: ignore
    CentralisedPolicyActor, CentralisedQValueCritic
):
    """Centralised multi-agent actor critic architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        shared_weights: bool = True,
    ):

        CentralisedQValueCritic.__init__(
            self,
            environment_spec=environment_spec,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            shared_weights=shared_weights,
        )


# TODO (Arnu): remove mypy type ignore once we can handle type checking for
# nested/multiple inheritance
class CentralisedSoftQValueActorCritic(  # type: ignore
    CentralisedSoftQValueCritic, CentralisedPolicyActor
):
    """Centralised multi-agent soft actor critic architecture."""

    def __init__(
        self,
        environment_spec: mava_specs.MAEnvironmentSpec,
        observation_networks: Dict[str, snt.Module],
        policy_networks: Dict[str, snt.Module],
        critic_V_networks: Dict[str, snt.Module],
        critic_Q_1_networks: Dict[str, snt.Module],
        critic_Q_2_networks: Dict[str, snt.Module],
        shared_weights: bool = True,
    ):

        CentralisedSoftQValueCritic.__init__(
            self,
            environment_spec=environment_spec,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            critic_V_networks=critic_V_networks,
            critic_Q_1_networks=critic_Q_1_networks,
            critic_Q_2_networks=critic_Q_2_networks,
            shared_weights=shared_weights,
        )
