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

"""Broadcasted communication for multi-agent RL systems"""

from typing import Dict

import sonnet as snt
import tensorflow as tf

from mava.components.tf.architectures import DecentralisedPolicyActor
from mava.components.tf.modules.communication import BaseCommunicationModule


class BroadcastedCommunication(BaseCommunicationModule):
    """Multi-agent broadcasted communication architecture."""

    def __init__(
        self,
        architecture: DecentralisedPolicyActor,
        shared: bool = True,
        channel_size: int = 4,
        channel_noise: float = 0.0,
    ) -> None:
        """Initializes the broadcaster communicator.
        Args:
            architecture: the BaseArchitecture used.
            shared: if a shared communication channel is used.
            channel_noise: stddev of normal noise in channel.
        """
        self._architecture = architecture
        self._shared = shared
        self._channel_noise = channel_noise

    def create_system(
        self,
    ) -> Dict[str, Dict[str, snt.Module]]:
        """Create system architecture with communication by modifying architecture."""

        return self._architecture.create_system()

    def create_behaviour_policy(self) -> Dict[str, snt.Module]:
        # Note (dries): Can't use the base architecture
        # because it assumes there is an observation network.
        return self._architecture._policy_networks

    def process_messages(
        self,
        messages: Dict[str, snt.Module],
    ) -> Dict[str, snt.Module]:
        """Initializes the broadcaster communicator.
        Args:
            messages: Dict of agent messages.
        """
        if self._shared:
            # Sum of all messages
            channel = tf.math.reduce_sum(tf.nest.flatten(messages), axis=0)
            # Add channel noise if applicable
            channel += tf.random.normal(
                channel.shape, mean=0.0, stddev=self._channel_noise
            )
        else:
            # Concat all messages (might need to retain ordering of dict)
            channel = tf.concat(tf.nest.flatten(messages), axis=0)
            # Add channel noise if applicable
            channel += tf.random.normal(
                channel.shape, mean=0.0, stddev=self._channel_noise
            )

        return {key: channel for key in messages}
