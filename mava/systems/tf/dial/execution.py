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

"""DIAL system executor implementation."""

from typing import Any, Dict, Optional

import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.systems.tf.madqn.execution import MADQNRecurrentCommExecutor
from mava.systems.tf.madqn.training import MADQNTrainer


class DIALSwitchExecutor(MADQNRecurrentCommExecutor):
    """DIAL executor.
    An executor based on a recurrent communicating policy for each agent in the system.
    Note: this executor is specific to switch game env.
    """

    def __init__(
        self,
        q_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        communication_module: BaseCommunicationModule,
        shared_weights: bool = True,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
        trainer: MADQNTrainer = None,
        fingerprint: bool = False,
        evaluator: bool = False,
    ):
        """Initialise the system executor

        Args:
            q_networks (Dict[str, snt.Module]): q-value networks for each agent in the
                system.
            action_selectors (Dict[str, Any]): policy action selector method, e.g.
                epsilon greedy.
            communication_module (BaseCommunicationModule): module for enabling
                communication protocols between agents.
            shared_weights (bool, optional): whether agents should share weights or not.
                Defaults to True.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            store_recurrent_state (bool, optional): boolean to store the recurrent
                network hidden state. Defaults to True.
            trainer (MADQNTrainer, optional): system trainer. Defaults to None.
            fingerprint (bool, optional): whether to use fingerprint stabilisation to
                stabilise experience replay. Defaults to False.
            evaluator (bool, optional): whether the executor will be used for
                evaluation. Defaults to False.
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._q_networks = q_networks
        self._policy_networks = q_networks
        self._communication_module = communication_module
        self._action_selectors = action_selectors
        self._store_recurrent_state = store_recurrent_state
        self._trainer = trainer
        self._shared_weights = shared_weights

        self._states: Dict[str, Any] = {}
        self._messages: Dict[str, Any] = {}

    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        state: types.NestedTensor,
        message: types.NestedTensor,
        legal_actions: types.NestedTensor,
        epsilon: tf.Tensor,
    ) -> types.NestedTensor:
        """Agent specific policy function

        Args:
            agent (str): agent id
            observation (types.NestedTensor): observation tensor received from the
                environment.
            state (types.NestedTensor): Recurrent network state.
            message (types.NestedTensor): received agent messsage.
            legal_actions (types.NestedTensor): actions allowed to be taken at the
                current observation.
            epsilon (tf.Tensor): value for epsilon greedy action selection.

        Returns:
            types.NestedTensor: action, message and new recurrent hidden state
        """

        (action, m_values), new_state = super()._policy(
            agent,
            observation,
            state,
            message,
            legal_actions,
            epsilon,
        )

        # Mask message if obs[0] == 1.
        # Note: this is specific to switch env
        if observation[0] == 0:
            m_values = tf.zeros_like(m_values)

        return (action, m_values), new_state
