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


"""Stabilising for multi-agent RL systems"""
from typing import Dict

import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils

from mava.components.tf.architectures import DecentralisedValueActor
from mava.components.tf.modules.stabilising import BaseStabilisationModule


class FingerPrintStabalisation(BaseStabilisationModule):
    """Multi-agent stabalisation architecture."""

    def __init__(
        self,
        architecture: DecentralisedValueActor,
    ) -> None:
        self._architecture = architecture
        self._fingerprint_spec = tf.ones((2,), dtype="float32")

    def create_actor_variables_with_fingerprints(
        self,
    ) -> Dict[str, Dict[str, snt.Module]]:

        actor_networks: Dict[str, Dict[str, snt.Module]] = {
            "values": {},
            "target_values": {},
        }

        # get actor specs
        actor_obs_specs = self._architecture._get_actor_specs()

        # create policy variables for each agent
        for agent_key in self._architecture._actor_agent_keys:

            obs_spec = actor_obs_specs[agent_key]

            # Create variables for value and policy networks.
            tf2_utils.create_variables(
                self._architecture._value_networks[agent_key],
                [obs_spec, self._fingerprint_spec],
            )

            # create target value network variables
            tf2_utils.create_variables(
                self._architecture._target_value_networks[agent_key],
                [obs_spec, self._fingerprint_spec],
            )

        actor_networks["values"] = self._architecture._value_networks
        actor_networks["target_values"] = self._architecture._target_value_networks

        return actor_networks

    def create_system(
        self,
    ) -> Dict[str, Dict[str, snt.Module]]:
        networks = self.create_actor_variables_with_fingerprints()
        return networks
