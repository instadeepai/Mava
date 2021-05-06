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

# TODO (StJohn): implement Qmix trainer
#   - Write code for training the mixing networks.
# Helper resources
#   - single agent dqn learner in acme:
#           https://github.com/deepmind/acme/blob/master/acme/agents/tf/dqn/learning.py
#   - multi-agent ddpg trainer in mava: mava/systems/tf/maddpg/trainer.py


"""Qmix trainer implementation."""

# import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import sonnet as snt
import tensorflow as tf
from acme.tf import savers as tf2_savers
from acme.utils import counting, loggers

import mava
from mava.utils import training_utils as train_utils


class QMIXTrainer(mava.Trainer):
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
        epsilon: tf.Variable,
        target_update_period: int,
        dataset: tf.data.Dataset,
        shared_weights: bool,
        optimizer: snt.Optimizer,
        clipping: bool,
        counter: counting.Counter,
        logger: loggers.Logger,
        checkpoint: bool,
    ) -> None:

        self._agents = agents
        self._agent_types = agent_types
        self._shared_weights = shared_weights
        self._optimizer = optimizer

        # Store online and target networks.
        self._q_networks = q_networks
        self._target_q_networks = target_q_networks
        self._mixing_network = mixing_network
        self._target_mixing_network = target_mixing_network
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

        # Global state (for hypernetwork) one-hot encoded
        s_tm1 = tf.one_hot(e_tm1["s_t"], depth=3)  # TODO Get depth from state specs
        s_t = tf.one_hot(e_t["s_t"], depth=3)

        # Do forward passes through the networks and calculate the losses
        with tf.GradientTape(persistent=True) as tape:
            # a_t = self._policy_actions(o_t_trans)

            q_tm1 = []  # Q vals
            q_t = []  # Target Q vals
            for agent in self._agents:
                agent_key = self.agent_net_keys[agent]

                o_tm1_feed, o_t_feed, a_tm1_feed = self._get_feed(
                    o_tm1, o_t, a_tm1, agent
                )

                # print("Batch state:", s_tm1, "\n") # NOTE Shouldn't these all be 0s?
                # print("Batch next state:", s_t, "\n")
                # print("Obs feed:", o_t_feed.observation, "\n")

                # [B, num_actions]
                q_tm1_agent = self._q_networks[agent_key](o_tm1_feed.observation)
                q_t_agent = self._target_q_networks[agent_key](o_t_feed.observation)

                # TODO Should I rather use policy to select my q_val than just max?
                q_tm1.append(q_tm1_agent)  # [B, num_actions] = [32,2]
                q_t.append(q_t_agent)  # Take only the best q_val

            # [B, num_actions*num_agents] = [32,4]
            q_tm1 = tf.concat(q_tm1, axis=1)
            q_t = tf.concat(q_t, axis=1)

            q_tot_mixed = self._mixing_network(q_tm1, s_tm1)  # [B, 1, 1]
            q_tot_target_mixed = self._target_mixing_network(q_t, s_t)  # [B, 1, 1]
            # print("Q mixed:", q_tot_mixed)

            # Cast the additional discount to match the environment discount dtype.
            # discount = tf.cast(self._discount, dtype=d_t.dtype)

            # Calculate Q loss.
            # Loss is MSE scaled by 0.5, so the gradient is equal to the TD error.
            discount = 0.99  # TODO Generalise

            # TODO Case where agents have different rewards?
            r_t = tf.reshape(r_t["agent_0"], shape=(-1, 1))
            # print("r_t.shape:",r_t.shape)
            # print("Q_t.shape:",tf.reduce_max(q_tot_target_mixed, axis=1).shape)
            target = tf.stop_gradient(
                r_t + discount * tf.reduce_max(q_tot_target_mixed, axis=1)
            )
            target = tf.reshape(target, (-1, 1, 1))
            # print("Target shape:",target.shape)
            td_error = target - q_tot_mixed
            # print("TD shape:", td_error.shape)

            self.loss = 0.5 * tf.square(td_error)
            # print(self.loss.shape)
            self.tape = tape

    def _backward(self) -> None:
        # Calculate the gradients and update the networks
        for agent in self._agents:
            agent_key = self.agent_net_keys[agent]
            # Get trainable variables.
            trainable_variables = self._q_networks[agent_key].trainable_variables

        trainable_variables += self._mixing_network.trainable_variables

        # Compute gradients.
        gradients = self.tape.gradient(self.loss, trainable_variables)

        # Maybe clip gradients.
        if self._clipping:
            gradients = tf.clip_by_global_norm(gradients, 40.0)[0]

        # Apply gradients.
        self._optimizer.apply(gradients, trainable_variables)

        # Delete the tape manually because of the persistent=True flag.
        train_utils.safe_del(self, "tape")

    def _step(
        self,
    ) -> Dict[str, Dict[str, Any]]:
        # Update the target networks
        # TODO Update target mixing network in this function
        self._update_target_networks()
        self._decrement_epsilon()

        inputs = next(self._iterator)

        self._forward(inputs)
        self._backward()

        return self.loss  # Return total system loss

    def step(self) -> None:
        # Run the learning step.
        # fetches = self._step()
        self._step()

        # Compute elapsed time.
        # timestamp = time.time()
        # if self._timestamp:
        #     elapsed_time = timestamp - self._timestamp
        # else:
        #     elapsed_time = 0
        # self._timestamp = timestamp  # type: ignore

        # Update our counts and record it.
        # counts = self._counter.increment(steps=1, walltime=elapsed_time)
        # fetches.update(counts)

        # Checkpoint and attempt to write the logs.

        # NOTE (Arnu): ignoring checkpointing and logging for now
        # self._checkpointer.save()
        # self._logger.write(fetches)

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        variables: Dict[str, Dict[str, np.ndarray]] = {}

        variables["mixing"] = self._mixing_network.variables  # Also hypernet vars
        variables["q_networks"] = {}  # or behaviour

        for key in self.unique_net_keys:
            variables["q_networks"][key] += self._q_networks[key].variables

        return variables
