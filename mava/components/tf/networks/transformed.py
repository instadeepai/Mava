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

# TODO (StJohn): complete class for transformed mixing
# https://github.com/oxwhirl/wqmix/blob/master/src/modules/mixers/qtran.py
# https://github.com/starry-sky6688/StarCraft/blob/master/network/qtran_net.py


"""Mixing for multi-agent RL systems"""

from typing import Optional, Tuple

import sonnet as snt
import tensorflow as tf


class QTranBase(snt.Module):
    """Multi-agent mixing architecture."""

    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        state_shape: Tuple,
        qtran_hidden_dim: int,
        rnn_hidden_dim: int,
        qtran_arch: str = "qtran_paper",
        larger_network: bool = True,
    ) -> None:
        """Initializes the mixer."""
        super(QTranBase, self).__init__()

        self._n_agents = n_agents
        self._n_actions = n_actions
        self._state_dim = int(tf.prod(state_shape))
        self._qtran_arch = qtran_arch
        self._qtran_hidden_dim = qtran_hidden_dim
        self._larger_network = larger_network
        self._rnn_hidden_dim = rnn_hidden_dim

        self._q_input_size = 0
        if self._qtran_arch == "coma_critic":
            self._q_input_size = self._state_dim + (self._n_agents * self._n_actions)
        elif self._qtran_arch == "qtran_paper":
            self._q_input_size = (
                self._state_dim + self._rnn_hidden_dim + self._n_actions
            )
        else:
            raise Exception(
                "{} is not a valid QTran architecture".format(self._qtran_arch)
            )

        # Define Q and V networks
        if self._larger_network:  # 3 hidden layers
            self._Q = snt.Sequential(
                [
                    snt.Flatten(preserve_dims=1),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(1),
                ]
            )
            self._V = snt.Sequential(
                [
                    snt.Flatten(preserve_dims=1),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(1),
                ]
            )
        else:  # 2 hidden layers
            self._Q = snt.Sequential(
                [
                    snt.Flatten(preserve_dims=1),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(1),
                ]
            )
            self._V = snt.Sequential(
                [
                    snt.Flatten(preserve_dims=1),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(1),
                ]
            )

        # Action encoding
        ae_input = self._rnn_hidden_dim + self._n_actions
        self._action_encoding = snt.Sequential(
            [
                snt.Flatten(preserve_dims=1),
                snt.Linear(ae_input),
                snt.relu(),
                snt.Linear(ae_input),
            ]
        )

    def __call__(
        self,
        batch: tf.Tensor,  # Convert this to whatever reverb uses.
        hidden_states: tf.Tensor,
        actions: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Perform mixing logic"""
        bs = batch.batch_size
        ts = batch.max_seq_length
        states = tf.reshape(batch["state"], shape=(bs * ts, self._state_dim))

        if actions is None:
            # Use the actions taken by the agents
            actions = tf.reshape(
                batch["actions_onehot"],
                shape=(bs * ts, self._n_agents * self._n_actions),
            )
        else:
            # It will arrive as (bs, ts, agents, actions), we need to reshape it
            actions = tf.reshape(
                actions, shape=(bs * ts, self._n_agents * self._n_actions)
            )

        if self._qtran_arch == "coma_critic":
            inputs = tf.concat((states, actions), axis=1)
        elif self._qtran_arch == "qtran_paper":
            hidden_states = tf.reshape(
                hidden_states, shape=(bs * ts, self._n_agents, -1)
            )
            agent_state_action_input = tf.concat((hidden_states, actions), axis=2)

            agent_state_action_encoding = self._action_encoding(
                tf.reshape(
                    agent_state_action_input, shape=(bs * ts * self._n_agents, -1)
                )
            )
            agent_state_action_encoding = tf.reshape(
                agent_state_action_encoding, shape=(bs * ts, self._n_agents, -1)
            )
            agent_state_action_encoding = tf.math.reduce_sum(
                agent_state_action_encoding, axis=1
            )  # Sum across agents

            inputs = tf.concat((states, agent_state_action_encoding), axis=1)

        q_outputs = self._Q(inputs)
        states = tf.reshape(batch["state"], shape=(bs * ts, self._state_dim))
        v_outputs = self._V(states)

        return q_outputs, v_outputs


class QTranAlt(snt.Module):
    def __init__(
        self,
        n_agents: int,
        n_actions: int,
        state_shape: Tuple,
        qtran_hidden_dim: int,
        rnn_hidden_dim: int,
        qtran_arch: str = "qtran_paper",
        larger_network: bool = True,
    ) -> None:
        """Initializes the mixer."""
        super(QTranAlt, self).__init__()

        self._n_agents = n_agents
        self._n_actions = n_actions
        self._state_dim = int(tf.prod(state_shape))
        self._qtran_arch = qtran_arch
        self._qtran_hidden_dim = qtran_hidden_dim
        self._larger_network = larger_network
        self._rnn_hidden_dim = rnn_hidden_dim

        # Q(s,-,u-i)
        # Q takes [state, u-i, i] as input
        self._q_input_size = self._state_dim + (self._n_agents * self._n_actions)

        # Define Q and V networks
        if self._larger_network:  # 3 hidden layers
            self._Q = snt.Sequential(
                [
                    snt.Flatten(preserve_dims=1),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(1),
                ]
            )
            self._V = snt.Sequential(
                [
                    snt.Flatten(preserve_dims=1),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(1),
                ]
            )
        else:  # 2 hidden layers
            self._Q = snt.Sequential(
                [
                    snt.Flatten(preserve_dims=1),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(1),
                ]
            )
            self._V = snt.Sequential(
                [
                    snt.Flatten(preserve_dims=1),
                    snt.Linear(self._qtran_hidden_dim),
                    tf.nn.relu(),
                    snt.Linear(1),
                ]
            )

    def __call__(
        self,
        batch: tf.Tensor,  # Convert this to whatever reverb uses.
        masked_actions: Optional[tf.Tensor] = None,
    ) -> tf.Tensor:
        """Perform mixing logic"""
        bs = batch.batch_size
        ts = batch.max_seq_length

        # NOTE (St John) Check this carefully. I'm not sure what exactly batch will
        # contain. My use of tf.tile() here is assuming tf is a tensor. Behaviour is
        # unchecked at this stage.

        # Repeat each state n_agents times
        repeated_states = tf.tile(batch["state"], multiples=[1, 1, self._n_agents])
        repeated_states = tf.reshape(repeated_states, shape=(-1, self._state_dim))

        if masked_actions is None:
            # Check what shape the incoming actions have when debugging.
            actions = tf.tile(
                batch["actions_onehot"], multiples=[1, 1, self._n_agents, 1]
            )

            agent_mask = 1 - tf.eye(self._n_agents)  # [n_agents,n_agents]
            agent_mask = tf.reshape(agent_mask, shape=(-1, 1))  # [n_agents^2,1]
            agent_mask = tf.tile(
                agent_mask, [1, self._n_actions]
            )  # [n_agents^2,n_actions]
            # Add extra dimensions for multiplication coming up
            # [1,1,n_agents^2,n_actions]
            agent_mask = tf.expand_dims(tf.expand_dims(agent_mask, 0), 0)

            masked_actions = actions * agent_mask
            masked_actions = tf.reshape(
                masked_actions, shape=(-1, self._n_agents * self._n_actions)
            )

        agent_ids = tf.eye(self._n_agents)  # [n_agents,n_agents]
        agent_ids = tf.expand_dims(
            tf.expand_dims(agent_ids, 0), 0
        )  # [1,1,n_agents,n_agents]
        agent_ids = tf.tile(agent_ids, [bs, ts, 1, 1])  # [bs,ts,n_agents,n_agents]
        agent_ids = tf.reshape(
            agent_ids, shape=(-1, self._n_agents)
        )  # [bs*ts*n_agents,n_agents]

        inputs = tf.concat([repeated_states, masked_actions, agent_ids], axis=1)

        q_outputs = self._Q(inputs)

        states = tf.tile(batch["state"], [1, 1, self._n_agents])
        states = tf.reshape(states, shape=(-1, self._state_dim))

        v_outputs = self._V(states)

        return q_outputs, v_outputs
