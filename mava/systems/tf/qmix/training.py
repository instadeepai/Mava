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

"""QMIX system trainer implementation."""

import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers
from trfl.indexing_ops import batched_index

from mava import types as mava_types
from mava.components.tf.modules.communication import BaseCommunicationModule
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
        agent_net_keys: Dict[str, str],
        checkpoint_minute_interval: int,
        exploration_scheduler: LinearExplorationScheduler,
        communication_module: Optional[BaseCommunicationModule] = None,
        max_gradient_norm: float = None,
        counter: counting.Counter = None,
        fingerprint: bool = False,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
    ) -> None:
        """Initialise QMIX trainer

        Args:
            agents (List[str]): agent ids, e.g. "agent_0".
            agent_types (List[str]): agent types, e.g. "speaker" or "listener".
            q_networks (Dict[str, snt.Module]): q-value networks.
            target_q_networks (Dict[str, snt.Module]): target q-value networks.
            mixing_network (snt.Module): mixing networks learning factorised q-value
                weights.
            target_mixing_network (snt.Module): target mixing networks.
            target_update_period (int): number of steps before updating target networks.
            dataset (tf.data.Dataset): training dataset.
            optimizer (Union[snt.Optimizer, Dict[str, snt.Optimizer]]): type of
                optimizer for updating the parameters of the networks.
            discount (float): discount factor for TD updates.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            checkpoint_minute_interval (int): The number of minutes to wait between
                checkpoints.
            exploration_scheduler (LinearExplorationScheduler): function specifying a
                decaying scheduler for epsilon exploration.
            communication_module (BaseCommunicationModule): module for communication
                between agents. Defaults to None.
            max_gradient_norm (float, optional): maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            counter (counting.Counter, optional): step counter object. Defaults to None.
            fingerprint (bool, optional): whether to apply replay stabilisation using
                policy fingerprints. Defaults to False.
            logger (loggers.Logger, optional): logger object for logging trainer
                statistics. Defaults to None.
            checkpoint (bool, optional): whether to checkpoint networks. Defaults to
                True.
            checkpoint_subpath (str, optional): subdirectory for storing checkpoints.
                Defaults to "~/mava/".
        """

        self._mixing_network = mixing_network
        self._target_mixing_network = target_mixing_network
        self._optimizer = optimizer

        super(QMIXTrainer, self).__init__(
            agents=agents,
            agent_types=agent_types,
            q_networks=q_networks,
            target_q_networks=target_q_networks,
            target_update_period=target_update_period,
            dataset=dataset,
            optimizer=optimizer,
            discount=discount,
            agent_net_keys=agent_net_keys,
            checkpoint_minute_interval=checkpoint_minute_interval,
            exploration_scheduler=exploration_scheduler,
            communication_module=communication_module,
            max_gradient_norm=max_gradient_norm,
            counter=counter,
            fingerprint=fingerprint,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

        # Checkpoint the mixing networks
        # TODO add checkpointing for mixing networks

    def _update_target_networks(self) -> None:
        """Sync the target network parameters with the latest online network
        parameters"""

        # Update target networks (incl. mixing networks).
        if tf.math.mod(self._num_steps, self._target_update_period) == 0:
            for key in self.unique_net_keys:
                online_variables = [
                    *self._q_networks[key].variables,
                ]
                target_variables = [
                    *self._target_q_networks[key].variables,
                ]

                # Make online -> target network update ops.
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(src)

            # NOTE These shouldn't really be in the agent for loop.
            online_variables = self.get_mixing_vars("mixing")
            target_variables = self.get_mixing_vars("target_mixing")

            # Make online -> target network update ops.
            for src, dest in zip(online_variables, target_variables):
                dest.assign(src)

        self._num_steps.assign_add(1)

    def step(self) -> None:
        """trainer step to update the parameters of the agents in the system"""

        # Run the learning step.
        fetches = self._step()

        # Compute elapsed time.
        timestamp = time.time()
        if self._timestamp:
            elapsed_time = timestamp - self._timestamp
        else:
            elapsed_time = 0
        self._timestamp = timestamp  # type: ignore

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        # Checkpoint and attempt to write the logs.
        if self._checkpoint:
            train_utils.checkpoint_networks(self._system_checkpointer)

        # Log and decrement epsilon
        epsilon = self.get_epsilon()
        fetches["epsilon"] = epsilon
        self._decrement_epsilon()

        if self._logger:
            self._logger.write(fetches)

    @tf.function
    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        """Trainer forward and backward passes."""

        # Update the target networks
        self._update_target_networks()

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)

        self._forward(inputs)

        self._backward()

        # Log losses per agent
        return {agent: {"q_value_loss": self.loss} for agent in self._agents}

    def _forward(self, inputs: reverb.ReplaySample) -> None:
        """Trainer forward pass

        Args:
            inputs (Any): input data from the data table (transitions)
        """

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
        trans = mava_types.Transition(*inputs.data)

        o_tm1, o_t, a_tm1, r_t, d_t, e_tm1, e_t = (
            trans.observations,
            trans.next_observations,
            trans.actions,
            trans.rewards,
            trans.discounts,
            trans.extras,
            trans.next_extras,
        )

        s_tm1 = e_tm1["s_t"]
        s_t = e_t["s_t"]

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            q_acts = []  # Q vals
            q_targets = []  # Target Q vals
            for agent in self._agents:
                agent_key = self._agent_net_keys[agent]

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

            q_tot_mixed = self._mixing_network(q_acts, s_tm1)  # [B, 1, 1]
            q_tot_target_mixed = self._target_mixing_network(
                q_targets, s_t
            )  # [B, 1, 1]

            # Calculate Q loss.
            targets = rewards + pcont * q_tot_target_mixed
            td_error = targets - q_tot_mixed

            # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
            self.loss = 0.5 * tf.reduce_mean(tf.square(td_error))
            self.tape = tape

    def _backward(self) -> None:
        """Trainer backward pass updating network parameters"""

        for agent in self._agents:
            agent_key = self._agent_net_keys[agent]
            # Update agent networks
            variables = self._q_networks[agent_key].trainable_variables
            gradients = self.tape.gradient(self.loss, variables)
            gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
            self._optimizers[agent_key].apply(gradients, variables)

        # Update mixing network
        variables = self.get_mixing_trainable_vars("mixing")
        gradients = self.tape.gradient(self.loss, variables)

        gradients = tf.clip_by_global_norm(gradients, self._max_gradient_norm)[0]
        self._optimizer.apply(gradients, variables)

        train_utils.safe_del(self, "tape")

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """get network variables

        Args:
            names (Sequence[str]): network names

        Returns:
            Dict[str, Dict[str, np.ndarray]]: network variables
        """

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

    def get_mixing_trainable_vars(self, network: str = "mixing") -> List:
        """get trainable mixing network variables

        Args:
            network (str, optional): mixing network name. Defaults to "mixing".

        Returns:
            List: network variables
        """

        mixing_trainable_vars = []
        for var in self._mixing_network.trainable_variables:
            if var.name.startswith(network):
                mixing_trainable_vars.append(var)
        return mixing_trainable_vars

    def get_mixing_vars(self, network: str = "mixing") -> List:
        """get all network variables, including target mixing variables.

        Args:
            network (str, optional): mixing network name. Defaults to "mixing".

        Returns:
            List: network variables
        """

        mixing_vars = []
        for var in self._mixing_network.variables:
            if var.name.startswith(network):
                mixing_vars.append(var)
        return mixing_vars
