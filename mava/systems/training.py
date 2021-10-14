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


"""MADDPG trainer implementation."""

from typing import Dict, List, Tuple

import reverb

from mava import types
from mava.core import SystemTrainer
from mava.callbacks import Callback
from mava.systems.callback_hook import SystemCallbackHookMixin


class Trainer(SystemTrainer, SystemCallbackHookMixin):
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

        self.on_training_init(self)

        self.on_training_init_end(self)

    def _update_target_networks(self) -> None:
        """Sync the target network parameters with the latest online network
        parameters"""

        self.on_training_update_target_networks_start(self)

        # for key in self.unique_net_keys:

        #     self.on_training_update_target_networks_get_variables(self, key)

        #     self.on_training_update_target_networks_update(self, key)

        self.on_training_update_target_networks(self)

        self.on_training_update_target_networks_end(self)

    def _transform_observations(
        self, obs: Dict[str, types.NestedArray], next_obs: Dict[str, types.NestedArray]
    ) -> Tuple:
        """Transform the observations using the observation networks of each agent."""

        self._obs = obs
        self._next_obs = next_obs

        self.on_training_transform_observations_start(self)

        self.on_training_transform_observations(self)

        self.on_training_transform_observations_end(self)

        self.transformed_observations

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

        self.feed

    def _step(self) -> Dict:
        """Trainer forward and backward passes."""

        self.on_training__step_start(self)

        self.on_training__step_update_target_networks(self)

        self.on_training__step_sample_batch(self)

        self.on_training__step_forward(self)

        self.on_training__step_backward(self)

        self.on_training__step_log(self)

        self.on_training__step_end(self)

        self.loss

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass"""

        self.on_training_forward_start(self)

        # self.on_training_forward_get_transitions(self)

        # with tf.GradientTape(persistent=True) as tape:

        #     self.on_training_forward_gradient_tape_start(self)

        #     for agent in self._trainer_agent_list:

        #         self.on_training_forward_agent_loop_start(self, agent)

        #         self.on_training_forward_agent_loop_get_feed(self, agent)

        self.on_training_forward(self)

        self.on_training_forward_end(self)

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

        self.on_training_backward_start(self)

        self.on_training_backward(self)

        self.on_training_backward_end(self)

    def step(self) -> None:
        """trainer step to update the parameters of the agents in the system"""

        self.on_training_step_start(self)

        self.on_training_step(self)

        self.on_training_step_end(self)
