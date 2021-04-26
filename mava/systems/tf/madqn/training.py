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

import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import sonnet as snt
import tensorflow as tf
import tree
import trfl
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava


class IDQNTrainer(mava.Trainer):
    """IDQN trainer.
    This is the trainer component of a MADDPG system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        q_networks: Dict[str, snt.Module],
        target_q_networks: Dict[str, snt.Module],
        observation_networks: Dict[str, snt.Module],
        epsilon: tf.Variable,
        target_update_period: int,
        dataset: tf.data.Dataset,
        shared_weights: bool,
        optimizer: snt.Optimizer = None,
        clipping: bool = True,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
    ):

        self._agents = agents
        self._agent_types = agent_types
        self._shared_weights = shared_weights
        self._optimizer = optimizer or snt.optimizers.Adam(1e-4)

        # Store online and target networks.
        self._q_networks = q_networks
        self._target_q_networks = target_q_networks
        self._observation_networks = observation_networks
        self._epsilon = epsilon

        # General learner book-keeping and loggers.
        self._counter = counter or counting.Counter()
        self._logger = logger or loggers.make_default_logger("trainer")

        # Other learner parameters.
        self._clipping = clipping

        # Necessary to track when to update target networks.
        self._num_steps = tf.Variable(0, dtype=tf.int32)
        self._target_update_period = target_update_period

        # Create an iterator to go through the dataset.
        self._iterator = iter(dataset)

        # Dictionary with network keys for each agent.
        self.agent_net_keys = {agent: agent for agent in self._agents}
        if self._shared_weights:
            self.agent_net_keys = {agent: agent.split("_")[0] for agent in self._agents}

        self.unique_net_keys = self._agent_types if shared_weights else self._agents

        # Checkpointer
        self._system_checkpointer = {}
        for agent_key in self.unique_net_keys:

            checkpointer = tf2_savers.Checkpointer(
                time_delta_minutes=5,
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

    @tf.function
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

    @tf.function
    def _transform_observations(
        self, state: Dict[str, np.ndarray], next_state: Dict[str, np.ndarray]
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
        o_tm1 = {}
        o_t = {}
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]
            o_tm1[agent] = self._observation_networks[agent_key](
                state[agent].observation
            )
            o_t[agent] = self._observation_networks[agent_key](
                next_state[agent].observation
            )
            # This stop_gradient prevents gradients to propagate into the target
            # observation network. In addition, since the online policy network is
            # evaluated at o_t, this also means the policy loss does not influence
            # the observation network training.
            o_t[agent] = tree.map_structure(tf.stop_gradient, o_t[agent])

        return o_tm1, o_t

    @tf.function
    def _get_feed(
        self,
        o_tm1_trans: Dict[str, np.ndarray],
        o_t_trans: Dict[str, np.ndarray],
        a_tm1: Dict[str, np.ndarray],
        agent: str,
    ) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:

        o_tm1_feed = o_tm1_trans[agent]
        o_t_feed = o_t_trans[agent]
        a_tm1_feed = a_tm1[agent]

        return o_tm1_feed, o_t_feed, a_tm1_feed

    def _decrement_epsilon(self) -> None:
        self._epsilon.assign_sub(1e-3)
        if self._epsilon < 0.01:
            self._epsilon.assign(0.01)

    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:

        # Update the target networks
        self._update_target_networks()

        # decrement epsilon
        self._decrement_epsilon()

        # Get data from replay (dropping extras if any). Note there is no
        # extra data here because we do not insert any into Reverb.
        inputs = next(self._iterator)

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
        o_tm1_trans, o_t_trans = self._transform_observations(o_tm1, o_t)

        logged_losses: Dict[str, Dict[str, Any]] = {}
        for agent in self._agents:

            agent_key = self.agent_net_keys[agent]

            with tf.GradientTape() as tape:
                # Maybe transform the observation before feeding into policy and critic.
                # Transforming the observations this way at the start of the learning
                # step effectively means that the policy and critic share observation
                # network weights.

                o_tm1_feed, o_t_feed, a_tm1_feed = self._get_feed(
                    o_tm1_trans, o_t_trans, a_tm1, agent
                )

                q_tm1 = self._q_networks[agent](o_tm1_feed)
                q_t = self._target_q_networks[agent](o_t_feed)

                loss, _ = trfl.qlearning(q_tm1, a_tm1_feed, r_t[agent], d_t[agent], q_t)

                loss = tf.reduce_mean(loss, axis=0)

            # Retrieve gradients
            q_network_variables = self._q_networks[agent_key].trainable_variables
            gradients = tape.gradient(loss, q_network_variables)

            # Maybe clip gradients.
            if self._clipping:
                gradients = tf.clip_by_global_norm(gradients, 40.0)[0]

            # Apply gradients.
            self._optimizer.apply(gradients, q_network_variables)

            logged_losses[agent] = {"loss": loss}

        return logged_losses

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

        # NOTE (Arnu): ignoring checkpointing and logging for now
        # self._checkpointer.save()
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
