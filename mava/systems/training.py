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


"""System Trainer implementation."""

from typing import Dict, List, Tuple

import reverb

from mava import types
from mava.core import SystemTrainer
from mava.callbacks import Callback, SystemCallbackHookMixin
from mava.utils import training_utils as train_utils


class Trainer(SystemTrainer):
    """MARL trainer"""

    def __init__(
        self,
        components: List[Callback] = [],
    ):
        """[summary]

        Args:
            components (List[Callback], optional): [description]. Defaults to [].
        """
        self.callbacks = components

        self.on_training_init_start(self)

        self.on_training_init_observation_networks(self)

        self.on_training_init_target_observation_networks(self)

        self.on_training_init_policy_networks(self)

        self.on_training_init_target_policy_networks(self)

        self.on_training_init_critic_networks(self)

        self.on_training_init_target_critic_networks(self)

        self.on_training_init_parameters(self)

        self.on_training_init(self)

        self.on_training_init_end(self)

    def _update_target_networks(self) -> None:
        """Sync the target network parameters with the latest online network
        parameters"""

        self.on_training_update_target_networks_start(self)

        for key in self.unique_net_keys:

            self._unique_net_key = key

            self.on_training_update_target_observation_networks(self)

            self.on_training_update_target_policy_networks(self)

            self.on_training_update_target_critic_networks(self)

        self.on_training_update_target_networks_end(self)

    def _transform_observations(
        self, obs: Dict[str, types.NestedArray], next_obs: Dict[str, types.NestedArray]
    ) -> Tuple:
        """Transform the observations using the observation networks of each agent."""

        self._obs = obs
        self._next_obs = next_obs

        self.on_training_transform_observations_start(self)

        for agent in self._agents:

            self._agent_key = self._agent_net_keys[agent]

            self.on_training_transform_observations(self)

            self.on_training_transform_target_observations(self)

        self.on_training_transform_observations_end(self)

        return self.transformed_observations

    def _get_feed(
        self,
        transition: Dict[str, Dict[str, types.NestedArray]],
        agent: str,
    ) -> Tuple:
        """get data to feed to the agent networks"""

        self._transition = transition
        self._agent = agent

        self.on_training_get_feed_start(self)

        self.on_training_get_feed(self)

        self.on_training_get_feed_end(self)

        return self.feed

    def _step(self) -> Dict:
        """Trainer forward and backward passes."""

        # Update the target networks
        self._update_target_networks()

        # Draw a batch of data from replay.
        sample: reverb.ReplaySample = next(self._iterator)

        self._forward(sample)

        self._backward()

        # Log losses per agent
        return train_utils.map_losses_per_agent_ac(*self.losses)