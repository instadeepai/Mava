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

import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import sonnet as snt
import tensorflow as tf
import tree
import trfl
from acme.tf import utils as tf2_utils
from acme.types import NestedArray
from acme.utils import counting, loggers

import mava
from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.systems.tf import savers as tf2_savers
from mava.utils import training_utils as train_utils


class MADQNTrainer(mava.Trainer):
    """MADQN trainer.
    This is the trainer component of a MADQN system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        q_networks: Dict[str, snt.Module],
        target_q_networks: Dict[str, snt.Module],
        target_update_period: int,
        dataset: tf.data.Dataset,
        optimizer: snt.Optimizer,
        discount: float,
        shared_weights: bool,
        exploration_scheduler: LinearExplorationScheduler,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
    ):

        self._agents = agents
        self._agent_types = agent_types
        self._shared_weights = shared_weights
        self._optimizer = optimizer
        self._checkpoint = checkpoint

        # Store online and target q-networks.
        self._q_networks = q_networks
        self._target_q_networks = target_q_networks

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger

        # Other learner parameters.
        self._discount = discount
        self._clipping = clipping

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_update_period = target_update_period

        # Create an iterator to go through the dataset.
        self._iterator = dataset

        # Store the exploration scheduler
        self._exploration_scheduler = exploration_scheduler

        # Dictionary with network keys for each agent.
        self.agent_net_keys = {agent: agent for agent in self._agents}
        if self._shared_weights:
            self.agent_net_keys = {agent: agent.split("_")[0] for agent in self._agents}

        self.unique_net_keys = self._agent_types if shared_weights else self._agents

        # Expose the variables.
        q_networks_to_expose = {}
        self._system_network_variables: Dict[str, Dict[str, snt.Module]] = {
            "q_network": {},
        }
        for agent_key in self.unique_net_keys:
            q_network_to_expose = self._target_q_networks[agent_key]

            q_networks_to_expose[agent_key] = q_network_to_expose

            self._system_network_variables["q_network"][
                agent_key
            ] = q_network_to_expose.variables

        # Checkpointer
        self._system_checkpointer = {}
        if checkpoint:
            for agent_key in self.unique_net_keys:

                checkpointer = tf2_savers.Checkpointer(
                    directory=checkpoint_subpath,
                    time_delta_minutes=15,
                    objects_to_save={
                        "counter": self._counter,
                        "q_network": self._q_networks[agent_key],
                        "target_q_network": self._target_q_networks[agent_key],
                        "optimizer": self._optimizer,
                        "num_steps": self._num_steps,
                    },
                    enable_checkpointing=checkpoint,
                )

                self._system_checkpointer[agent_key] = checkpointer

        # Do not record timestamps until after the first learning step is done.
        # This is to avoid including the time it takes for actors to come online and
        # fill the replay buffer.

        self._timestamp = None

    def get_epsilon(self) -> float:
        epsilon = self._exploration_scheduler.get_epsilon()
        return epsilon

    def _decrement_epsilon(self) -> None:
        self._exploration_scheduler.decrement_epsilon()

    def _update_target_networks(self) -> None:
        for key in self.unique_net_keys:
            # Update target network.
            online_variables = (*self._q_networks[key].variables,)

            target_variables = (*self._target_q_networks[key].variables,)

            # Make online -> target network update ops.
            if tf.math.mod(self._num_steps, self._target_update_period) == 0:
                for src, dest in zip(online_variables, target_variables):
                    dest.assign(src)
        self._num_steps.assign_add(1)

    def _get_feed(
        self,
        o_tm1_trans: Dict[str, np.ndarray],
        o_t_trans: Dict[str, np.ndarray],
        a_tm1: Dict[str, np.ndarray],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        # Decentralised
        o_tm1_feed = o_tm1_trans[agent].observation
        o_t_feed = o_t_trans[agent].observation
        a_tm1_feed = a_tm1[agent]

        return o_tm1_feed, o_t_feed, a_tm1_feed

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
        return self._q_network_losses

    def _forward(self, inputs: Any) -> None:
        # Unpack input data as follows:
        # o_tm1 = dictionary of observations one for each agent
        # a_tm1 = dictionary of actions taken from obs in o_tm1
        # r_t = dictionary of rewards or rewards sequences
        #   (if using N step transitions) ensuing from actions a_tm1
        # d_t = environment discount ensuing from actions a_tm1.
        #   This discount is applied to future rewards after r_t.
        # o_t = dictionary of next observations or next observation sequences
        # e_t [Optional] = extra data that the agents persist in replay.
        o_tm1, a_tm1, _, r_t, d_t, o_t, _ = inputs.data

        with tf.GradientTape(persistent=True) as tape:
            q_network_losses: Dict[str, NestedArray] = {}

            for agent in self._agents:
                agent_key = self.agent_net_keys[agent]

                # Cast the additional discount to match the environment discount dtype.
                discount = tf.cast(self._discount, dtype=d_t[agent].dtype)

                # Maybe transform the observation before feeding into policy and critic.
                # Transforming the observations this way at the start of the learning
                # step effectively means that the policy and critic share observation
                # network weights.

                o_tm1_feed, o_t_feed, a_tm1_feed = self._get_feed(
                    o_tm1, o_t, a_tm1, agent
                )

                # Q-network learning
                q_tm1 = self._q_networks[agent_key](o_tm1_feed)
                q_t = self._target_q_networks[agent_key](o_t_feed)

                loss, _ = trfl.qlearning(
                    q_tm1, a_tm1_feed, r_t[agent], discount * d_t[agent], q_t
                )

                loss = tf.reduce_mean(loss)

                q_network_losses[agent] = {}
                q_network_losses[agent]["q_value_loss"] = loss

        self._q_network_losses = q_network_losses
        self.tape = tape

    def _backward(self) -> None:
        q_network_losses = self._q_network_losses
        tape = self.tape
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]

            # Get trainable variables
            q_network_variables = self._q_networks[agent_key].trainable_variables

            # Compute gradients
            gradients = tape.gradient(q_network_losses[agent], q_network_variables)

            # Maybe clip gradients.
            if self._clipping:
                gradients = tf.clip_by_global_norm(gradients, 40.0)[0]

            # Apply gradients.
            self._optimizer.apply(gradients, q_network_variables)

        train_utils.safe_del(self, "tape")

    def step(self) -> None:
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

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        variables: Dict[str, Dict[str, np.ndarray]] = {}
        for network_type in names:
            variables[network_type] = {}
            for agent in self.unique_net_keys:
                variables[network_type][agent] = tf2_utils.to_numpy(
                    self._system_network_variables[network_type][agent]
                )
        return variables


class RecurrentMADQNTrainer(MADQNTrainer):
    """Recurrent MADQN trainer.
    This is the trainer component of a MADQN system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        q_networks: Dict[str, snt.Module],
        target_q_networks: Dict[str, snt.Module],
        target_update_period: int,
        dataset: tf.data.Dataset,
        optimizer: snt.Optimizer,
        discount: float,
        shared_weights: bool,
        exploration_scheduler: LinearExplorationScheduler,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
    ):
        super().__init__(
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
            clipping=clipping,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
        )

    def _forward(self, inputs: Any) -> None:
        data = tree.map_structure(
            lambda v: tf.expand_dims(v, axis=0) if len(v.shape) <= 1 else v, inputs.data
        )
        data = tf2_utils.batch_to_sequence(data)

        observations, actions, rewards, discounts, _, extra = data

        # core_states = extra["core_states"]
        core_state = tree.map_structure(
            lambda s: s[:, 0, :], inputs.data.extras["core_states"]
        )

        with tf.GradientTape(persistent=True) as tape:
            q_network_losses: Dict[str, NestedArray] = {}

            for agent in self._agents:
                agent_key = self.agent_net_keys[agent]
                # Cast the additional discount to match the environment discount dtype.
                discount = tf.cast(self._discount, dtype=discounts[agent][0].dtype)

                q, s = snt.static_unroll(
                    self._q_networks[agent_key],
                    observations[agent].observation,
                    core_state[agent][0],
                )

                q_targ, s = snt.static_unroll(
                    self._target_q_networks[agent_key],
                    observations[agent].observation,
                    core_state[agent][0],
                )

                q_network_losses[agent] = {}
                q_network_losses[agent]["q_value_loss"] = tf.zeros(())

                for t in range(1, q.shape[0]):
                    loss, _ = trfl.qlearning(
                        q[t - 1],
                        actions[agent][t - 1],
                        rewards[agent][t],
                        discount * discounts[agent][t],
                        q_targ[t],
                    )

                    loss = tf.reduce_mean(loss)
                    q_network_losses[agent]["q_value_loss"] += loss

        self._q_network_losses = q_network_losses
        self.tape = tape
