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

"""VDN trainer implementation."""

from typing import Any, Dict, List, Optional, Union

import sonnet as snt
import tensorflow as tf
from acme.utils import counting, loggers
from trfl.indexing_ops import batched_index

from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.systems.tf.madqn.training import MADQNTrainer
from mava.utils import training_utils as train_utils

train_utils.set_growing_gpu_memory()


class VDNTrainer(MADQNTrainer):
    """VDN trainer.
    This is the trainer component of a VDN system. i.e. it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        q_networks: Dict[str, snt.Module],
        target_q_networks: Dict[str, snt.Module],
        mixing_network: snt.Module,
        target_mixing_network: snt.Module,
        target_update_period: int,
        dataset: tf.data.Dataset,
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        discount: float,
        shared_weights: bool,
        exploration_scheduler: LinearExplorationScheduler,
        communication_module: Optional[BaseCommunicationModule] = None,
        max_gradient_norm: float = None,
        counter: counting.Counter = None,
        fingerprint: bool = False,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
    ) -> None:

        self._mixing_network = mixing_network
        self._target_mixing_network = target_mixing_network

        super(VDNTrainer, self).__init__(
            agents=agents,
            agent_types=agent_types,
            q_networks=q_networks,
            target_q_networks=target_q_networks,
            target_update_period=target_update_period,
            dataset=dataset,
            optimizer=optimizer,
            discount=discount,
            shared_weights=shared_weights,
            exploration_scheduler=exploration_scheduler,
            communication_module=communication_module,
            max_gradient_norm=max_gradient_norm,
            counter=counter,
            fingerprint=fingerprint,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

    @tf.function
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        # Update the target networks
        self._update_target_networks()

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)

        self._forward(inputs)

        self._backward()

        # Log losses per agent
        return {agent: {"q_value_loss": self.loss} for agent in self._agents}

    def _forward(self, inputs: Any) -> None:
        # Unpack input data as follows:
        # o_tm1 = dictionary of observations one for each agent
        # a_tm1 = dictionary of actions taken from obs in o_tm1
        # e_tm1 [Optional] = extra data that the agents persist in replay.
        # r_t = dictionary of rewards or rewards sequences
        #   (if using N step transitions) ensuing from actions a_tm1
        # d_t = environment discount ensuing from actions a_tm1.
        #   This discount is applied to future rewards after r_t.
        # o_t = dictionary of next observations or next observation sequences
        # e_t = [Optional] = extra data that the agents persist in replay.
        o_tm1, a_tm1, _, r_t, d_t, o_t, _ = inputs.data

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            q_acts = []  # Q vals
            q_targets = []  # Target Q vals
            for agent in self._agents:
                agent_key = self.agent_net_keys[agent]

                o_tm1_feed, o_t_feed, a_tm1_feed = self._get_feed(
                    o_tm1, o_t, a_tm1, agent
                )
                q_tm1 = self._q_networks[agent_key](o_tm1_feed)
                q_t_value = self._target_q_networks[agent_key](o_t_feed)
                q_t_selector = self._q_networks[agent_key](o_t_feed)
                best_action = tf.argmax(q_t_selector, axis=1, output_type=tf.int32)

                # TODO Make use of q_t_selector for fingerprinting. Speak to Claude.
                q_act = batched_index(q_tm1, a_tm1_feed, keepdims=True)  # [B, 1]
                q_target = batched_index(
                    q_t_value, best_action, keepdims=True
                )  # [B, 1]

                q_acts.append(q_act)
                q_targets.append(q_target)

            rewards = tf.concat(
                [tf.reshape(val, (-1, 1)) for val in list(r_t.values())], axis=1
            )
            rewards = tf.reduce_mean(rewards, axis=1)  # [B]

            pcont = tf.concat(
                [tf.reshape(val, (-1, 1)) for val in list(d_t.values())], axis=1
            )
            pcont = tf.reduce_mean(pcont, axis=1)
            discount = tf.cast(self._discount, list(d_t.values())[0].dtype)
            pcont = discount * pcont  # [B]

            q_acts = tf.concat(q_acts, axis=1)  # [B, num_agents]
            q_targets = tf.concat(q_targets, axis=1)  # [B, num_agents]

            q_tot_mixed = self._mixing_network(q_acts)  # [B, 1, 1]
            q_tot_target_mixed = self._target_mixing_network(q_targets)  # [B, 1, 1]

            # q_tot_mixed = tf.reduce_sum(q_acts, axis=1)  # [B, 1, 1]
            # q_tot_target_mixed = tf.reduce_sum(q_targets, axis=1)  # [B, 1, 1]

            # Calculate Q loss.
            targets = rewards + pcont * q_tot_target_mixed
            targets = tf.stop_gradient(targets)
            td_error = targets - q_tot_mixed

            # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
            self.loss = 0.5 * tf.reduce_mean(tf.square(td_error))
            self.tape = tape

    def _backward(self) -> None:
        # Calculate the gradients and update the networks
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]
            # Get trainable variables.
            trainable_variables = self._q_networks[agent_key].trainable_variables

            # Compute gradients.
            gradients = self.tape.gradient(self.loss, trainable_variables)

            # Clip gradients.
            gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

            # Apply gradients.
            self._optimizers[agent_key].apply(gradients, trainable_variables)

        # Delete the tape manually because of the persistent=True flag.
        train_utils.safe_del(self, "tape")
