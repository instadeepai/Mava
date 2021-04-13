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

from mava.components.tf.architectures import BaseArchitecture
from mava.components.tf.modules.mixing import BaseMixingModule


class QTranBase(BaseMixingModule):
    """Multi-agent mixing architecture."""

    def __init__(
        self,
        architecture: BaseArchitecture,
        n_agents: int,
        n_actions: int,
        # Choice of underlying architecture.
        # Can this be derived from Mava 'architecture' parameter?
        state_shape: Tuple,
        qtran_hidden_dim: int,
        rnn_hidden_dim: int,
        qtran_arch: str = "qtran_paper",
        larger_network: bool = True,
    ) -> None:
        """Initializes the mixer."""
        super(QTranBase, self).__init__()
        self._architecture = architecture

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
        states = batch["state"].reshape(bs * ts, self._state_dim)

        if actions is None:
            # Use the actions taken by the agents
            actions = snt.reshape(
                batch["actions_onehot"],
                output_shape=(bs * ts, self._n_agents * self._n_actions),
            )
        else:
            # It will arrive as (bs, ts, agents, actions), we need to reshape it
            actions = snt.reshape(
                actions, output_shape=(bs * ts, self._n_agents * self._n_actions)
            )

        if self._qtran_arch == "coma_critic":
            inputs = tf.concat((states, actions), axis=1)
        elif self._qtran_arch == "qtran_paper":
            hidden_states = snt.reshape(
                hidden_states, output_shape=(bs * ts, self.n_agents, -1)
            )
            agent_state_action_input = tf.concat((hidden_states, actions), axis=2)

            agent_state_action_encoding = self._action_encoding(
                snt.reshape(
                    agent_state_action_input, output_shape=(bs * ts * self.n_agents, -1)
                )
            )
            agent_state_action_encoding = snt.reshape(
                agent_state_action_encoding, output_shape=(bs * ts, self.n_agents, -1)
            )
            agent_state_action_encoding = tf.math.reduce_sum(
                agent_state_action_encoding, axis=1
            )  # Sum across agents

            inputs = tf.concat((states, agent_state_action_encoding), axis=1)

        q_outputs = self._Q(inputs)
        states = snt.reshape(batch["state"], output_shape=(bs * ts, self.state_dim))
        v_outputs = self._V(states)

        return q_outputs, v_outputs


class QTranAlt(BaseMixingModule):
    def __init__(self, architecture: BaseArchitecture) -> None:
        super(QTranAlt, self).__init__()
        self._architecture = architecture

    def __call__(self) -> None:
        """Perform mixing logic"""
