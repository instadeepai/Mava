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


"""DIAL system trainer implementation."""

from typing import Any, Dict, List, Union

import sonnet as snt
import tensorflow as tf
import tree
import trfl
from acme.tf import utils as tf2_utils
from acme.types import NestedArray
from acme.utils import counting, loggers

from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.systems.tf.madqn.training import MADQNRecurrentCommTrainer
from mava.utils import training_utils as train_utils

train_utils.set_growing_gpu_memory()


class DIALSwitchTrainer(MADQNRecurrentCommTrainer):
    """Recurrent Comm DIAL Switch trainer.
    This is the trainer component of a DIAL system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    Note: this trainer is specific to switch game env.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        q_networks: Dict[str, snt.Module],
        target_q_networks: Dict[str, snt.Module],
        target_update_period: int,
        dataset: tf.data.Dataset,
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]],
        discount: float,
        agent_net_keys: Dict[str, str],
        checkpoint_minute_interval: int,
        exploration_scheduler: LinearExplorationScheduler,
        communication_module: BaseCommunicationModule,
        max_gradient_norm: float = None,
        fingerprint: bool = False,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
    ):
        """Initialise DIAL trainer for switch game

        Args:
            agents (List[str]): agent ids, e.g. "agent_0".
            agent_types (List[str]): agent types, e.g. "speaker" or "listener".
            q_networks (Dict[str, snt.Module]): q-value networks.
            target_q_networks (Dict[str, snt.Module]): target q-value networks.
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
                between agents.
            max_gradient_norm (float, optional): maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            fingerprint (bool, optional): whether to apply replay stabilisation using
                policy fingerprints. Defaults to False.
            counter (counting.Counter, optional): step counter object. Defaults to None.
            logger (loggers.Logger, optional): logger object for logging trainer
                statistics. Defaults to None.
            checkpoint (bool, optional): whether to checkpoint networks. Defaults to
                True.
            checkpoint_subpath (str, optional): subdirectory for storing checkpoints.
                Defaults to "~/mava/".
        """

        super().__init__(
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
            max_gradient_norm=max_gradient_norm,
            fingerprint=fingerprint,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
            communication_module=communication_module,
        )

    def _forward(self, inputs: Any) -> None:
        data = tree.map_structure(
            lambda v: tf.expand_dims(v, axis=0) if len(v.shape) <= 1 else v, inputs.data
        )
        data = tf2_utils.batch_to_sequence(data)

        observations, actions, rewards, discounts, _, _ = (
            data.observations,
            data.actions,
            data.rewards,
            data.discounts,
            data.start_of_episode,
            data.extras,
        )

        # Using extra directly from inputs due to shape.
        core_state = tree.map_structure(
            lambda s: s[:, 0, :], inputs.data.extras["core_states"]
        )
        core_message = tree.map_structure(
            lambda s: s[:, 0, :], inputs.data.extras["core_messages"]
        )
        T = actions[self._agents[0]].shape[0]

        # Use fact that end of episode always has the reward to
        # find episode lengths. This is used to mask loss.
        ep_end = tf.argmax(tf.math.abs(rewards[self._agents[0]]), axis=0)

        with tf.GradientTape(persistent=True) as tape:
            q_network_losses: Dict[str, NestedArray] = {
                agent: {"q_value_loss": tf.zeros(())} for agent in self._agents
            }

            state = {agent: core_state[agent][0] for agent in self._agents}
            target_state = {agent: core_state[agent][0] for agent in self._agents}

            message = {agent: core_message[agent][0] for agent in self._agents}
            target_message = {agent: core_message[agent][0] for agent in self._agents}

            # _target_q_networks must be 1 step ahead
            target_channel = self._communication_module.process_messages(target_message)
            for agent in self._agents:
                agent_key = self._agent_net_keys[agent]
                (q_targ, m), s = self._target_q_networks[agent_key](
                    observations[agent].observation[0],
                    target_state[agent],
                    target_channel[agent],
                )
                target_state[agent] = s
                target_message[agent] = m

            for t in range(1, T, 1):
                channel = self._communication_module.process_messages(message)
                target_channel = self._communication_module.process_messages(
                    target_message
                )

                for agent in self._agents:
                    agent_key = self._agent_net_keys[agent]

                    # Cast the additional discount
                    # to match the environment discount dtype.

                    discount = tf.cast(self._discount, dtype=discounts[agent][0].dtype)

                    (q_targ, m), s = self._target_q_networks[agent_key](
                        observations[agent].observation[t],
                        target_state[agent],
                        target_channel[agent],
                    )

                    target_state[agent] = s
                    target_message[agent] = tf.math.multiply(
                        m, observations[agent].observation[t][:, :1]
                    )

                    (q, m), s = self._q_networks[agent_key](
                        observations[agent].observation[t - 1],
                        state[agent],
                        channel[agent],
                    )

                    state[agent] = s
                    message[agent] = tf.math.multiply(
                        m, observations[agent].observation[t - 1][:, :1]
                    )

                    # Mask target
                    q_targ = tf.concat(
                        [
                            [q_targ[i]]
                            if t <= ep_end[i]
                            else [tf.zeros_like(q_targ[i])]
                            for i in range(q_targ.shape[0])
                        ],
                        axis=0,
                    )

                    loss, _ = trfl.qlearning(
                        q,
                        actions[agent][t - 1],
                        rewards[agent][t - 1],
                        discount * discounts[agent][t],
                        q_targ,
                    )

                    # Index loss (mask ended episodes)
                    if not tf.reduce_any(t - 1 <= ep_end):
                        continue

                    loss = tf.reduce_mean(loss[t - 1 <= ep_end])
                    # loss = tf.reduce_mean(loss)
                    q_network_losses[agent]["q_value_loss"] += loss

        self._q_network_losses = q_network_losses
        self.tape = tape
