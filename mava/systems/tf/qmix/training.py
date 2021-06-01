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

"""Qmix trainer implementation."""

from typing import Any, Dict, List, Sequence

import numpy as np
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers
from trfl.indexing_ops import batched_index

from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)

# from mava.systems.tf import savers as tf2_savers
from mava.systems.tf.madqn.training import MADQNTrainer
from mava.utils import training_utils as train_utils

train_utils.set_growing_gpu_memory()


class QMIXTrainer(MADQNTrainer):
    """QMIX trainer.
    This is the trainer component of a QMIX system. i.e. it takes a dataset as input
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
        optimizer: snt.Optimizer,
        discount: float,
        shared_weights: bool,
        exploration_scheduler: LinearExplorationScheduler,
        max_gradient_norm: float = None,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
    ) -> None:

        self._mixing_network = mixing_network
        self._target_mixing_network = target_mixing_network

        super(QMIXTrainer, self).__init__(
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
            max_gradient_norm=max_gradient_norm,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

        # Checkpoint the mixing networks
        # if checkpoint:
        # TODO Checkpointing of mixing networks not playing nicely...
        # self._system_checkpointer["mixing_network"] = tf2_savers.Checkpointer(
        #     directory=checkpoint_subpath,
        #     time_delta_minutes=15,
        #     objects_to_save={
        #         "mixing_network": self._mixing_network,
        #     },
        #     enable_checkpointing=checkpoint,
        # )
        # self._system_checkpointer[
        #     "target_mixing_network"
        # ] = tf2_savers.Checkpointer(
        #     directory=checkpoint_subpath,
        #     time_delta_minutes=15,
        #     objects_to_save={
        #         "target_mixing_network": self._target_mixing_network,
        #     },
        #     enable_checkpointing=checkpoint,
        # )

    def _update_target_networks(self) -> None:
        online_variables = []
        target_variables = []
        for key in self.unique_net_keys:
            # Update target networks (incl. mixing networks).
            online_variables += self._q_networks[key].variables
            target_variables += self._target_q_networks[key].variables

        online_variables += self._mixing_network.variables
        target_variables += self._target_mixing_network.variables

        # Make online -> target network update ops.
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)

        self._num_steps.assign_add(1)

    @tf.function
    def _step(self) -> Dict[str, Dict[str, Any]]:
        # Update the target networks
        self._update_target_networks()

        inputs = next(self._iterator)

        self._forward(inputs)
        self._backward()

        return {"system": {"q_value_loss": self.loss}}  # Return total system loss

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
        o_tm1, a_tm1, e_tm1, r_t, d_t, o_t, e_t = inputs.data

        # Global state (for hypernetwork)
        # TODO Switch between one-hot for discrete state space and no one-hot
        # for continuous state spaces. `depth` param should be taken from
        # state specs - or somewhere else in a cleaner way?

        # One-hot if discrete states
        # s_tm1 = tf.one_hot(e_tm1["s_t"], depth=3)
        # s_t = tf.one_hot(e_t["s_t"], depth=3)

        # Don't one-hot for continuous states
        s_tm1 = e_tm1["s_t"]
        s_t = e_t["s_t"]

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            q_tm1 = []  # Q vals
            q_t = []  # Target Q vals
            for agent in self._agents:
                agent_key = self.agent_net_keys[agent]

                o_tm1_feed, o_t_feed, a_tm1_feed = self._get_feed(
                    o_tm1, o_t, a_tm1, agent
                )

                q_tm1_agent = self._q_networks[agent_key](o_tm1_feed)  # [B, n_actions]
                q_act = batched_index(q_tm1_agent, a_tm1_feed, keepdims=True)  # [B, 1]

                q_t_agent = self._target_q_networks[agent_key](
                    o_t_feed
                )  # [B, n_actions]
                q_target_max = tf.reduce_max(q_t_agent, axis=1, keepdims=True)  # [B, 1]

                q_tm1.append(q_act)
                q_t.append(q_target_max)

            num_agents = len(self._agents)

            rewards = [tf.reshape(val, (-1, 1)) for val in list(r_t.values())]
            rewards = tf.reshape(
                tf.concat(rewards, axis=1), (-1, 1, num_agents)
            )  # [B, 1, num_agents]

            dones = [tf.reshape(val.terminal, (-1, 1)) for val in list(o_tm1.values())]
            dones = tf.reshape(
                tf.concat(dones, axis=1), (-1, 1, num_agents)
            )  # [B, 1, num_agents]

            q_tm1 = tf.concat(q_tm1, axis=1)  # [B, num_agents]
            q_t = tf.concat(q_t, axis=1)  # [B, num_agents]

            q_tot_mixed = self._mixing_network(q_tm1, s_tm1)  # [B, 1, 1]
            q_tot_target_mixed = self._target_mixing_network(q_t, s_t)  # [B, 1, 1]

            # Calculate Q loss.
            targets = (
                rewards
                + self._discount * (tf.constant(1.0) - dones) * q_tot_target_mixed
            )
            targets = tf.stop_gradient(targets)
            td_error = targets - q_tot_mixed

            # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
            self.loss = 0.5 * tf.reduce_mean(tf.square(td_error))
            self.tape = tape

    def _backward(self) -> None:
        # Calculate the gradients and update the networks
        trainable_variables = []
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]
            # Get trainable variables.
            trainable_variables += self._q_networks[agent_key].trainable_variables

        trainable_variables += self._mixing_network.trainable_variables

        # Compute gradients.
        gradients = self.tape.gradient(self.loss, trainable_variables)

        # Clip gradients.
        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]

        # Apply gradients.
        self._optimizers[agent_key].apply(gradients, trainable_variables)

        # Delete the tape manually because of the persistent=True flag.
        train_utils.safe_del(self, "tape")

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        variables: Dict[str, Dict[str, np.ndarray]] = {}
        variables = {}
        for network_type in names:
            if network_type == "mixing":
                # Includes the hypernet variables
                variables[network_type] = self._mixing_network.variables
            else:  # Collect variables for each agent network
                variables[network_type] = {
                    key: tf2_utils.to_numpy(
                        self._system_network_variables[network_type][key]
                    )
                    for key in self.unique_net_keys
                }
        return variables
