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


"""TF system trainer implementation."""

import reverb

from mava.systems.training import Trainer
from mava.utils import training_utils as train_utils


class TFTrainer(Trainer):
    """MARL trainer"""

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass"""

        self._inputs = inputs

        self.on_training_forward_start(self)

        self.on_training_forward_get_transitions(self)

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:

            self.on_training_forward_tape_start(self)

            for agent in self._trainer_agent_list:
                self._agent_key = self._agent_net_keys[agent]

                self._get_feed(self._transition)

                self.on_training_forward_tape_critic_loss(self)

                self.on_training_forward_tape_actor_loss(self)

            self.on_training_forward_tape_end(self)

        self._tape = tape

        self.on_training_forward_end(self)

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

        self.on_training_backward_start(self)

        for agent in self._trainer_agent_list:
            self._agent_key = self._agent_net_keys[agent]

            self.on_training_backwards_get_observation_variables(self)
            self.on_training_backwards_get_policy_variables(self)
            self.on_training_backwards_get_critic_variables(self)

            self.on_training_backwards_compute_observation_gradients(self)
            self.on_training_backwards_compute_policy_gradients(self)
            self.on_training_backwards_compute_critic_gradients(self)

            self.on_training_backwards_update_observation_parameters(self)
            self.on_training_backwards_update_policy_parameters(self)
            self.on_training_backwards_update_critic_parameters(self)

        self.on_training_backward_end(self)

        train_utils.safe_del(self, "_tape")