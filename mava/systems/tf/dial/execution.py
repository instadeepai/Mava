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

# TODO (Kevin): implement DIAL executor (if required)
# Helper resources
#   - single agent generic actors in acme:
#           https://github.com/deepmind/acme/blob/master/acme/agents/tf/actors.py
#   - single agent custom actor for Impala in acme:
#           https://github.com/deepmind/acme/blob/master/acme/agents/tf/impala/acting.py
#   - multi-agent generic executors in mava: mava/systems/tf/executors.py

"""DIAL executor implementation."""
from typing import Dict, Optional

import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme import types
from acme.specs import EnvironmentSpec
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.systems.tf.executors import RecurrentExecutorWithComms

tfd = tfp.distributions


class DIALExecutor(RecurrentExecutorWithComms):
    """DIAL implementation of a recurrent Executor."""

    def __init__(
        self,
        policy_networks: Dict[str, snt.RNNCore],
        communication_module: BaseCommunicationModule,
        agent_specs: Dict[str, EnvironmentSpec],
        shared_weights: bool = True,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
        is_eval: bool = False,
        epsilon: float = 0.05,
    ):
        """Initializes the executor.
        TODO: Update docstring
        Args:
          policy_networks: the (recurrent) policy to run for each agent in the system.
          shared_weights: specify if weights are shared between agent networks.
          adder: the adder object to which allows to add experiences to a
            dataset/replay buffer.
          variable_client: object which allows to copy weights from the trainer copy
            of the policies to the executor copy (in case they are separate).
          store_recurrent_state: Whether to pass the recurrent state to the adder.
        """
        # Store these for later use.
        self._agent_specs = agent_specs
        self._is_eval = is_eval
        self._epsilon = epsilon

        super().__init__(
            adder=adder,
            variable_client=variable_client,
            policy_networks=policy_networks,
            store_recurrent_state=store_recurrent_state,
            communication_module=communication_module,
            shared_weights=shared_weights,
        )

    def _sample_action(
        self, action_policy: types.NestedTensor, agent: str
    ) -> types.NestedTensor:
        action = tf.argmax(action_policy, axis=1)
        if tf.random.uniform([]) < self._epsilon and not self._is_eval:
            action_spec = self._agent_specs[agent].actions
            action = tf.random.uniform(
                action_spec.shape, 0, action_spec.num_values, dtype=tf.dtypes.int64
            )

        # Hard coded perfect policy:
        # if observation[1].item()==5 and observation[0].item()==1:
        #   action = tf.constant([1], dtype=tf.dtypes.int64)
        # else:
        #   tf.constant([0], dtype=tf.dtypes.int64)

        return action

    def _process_message(
        self, observation: types.NestedTensor, message_policy: types.NestedTensor
    ) -> types.NestedTensor:
        # Only one agent can message at each timestep
        if observation[0] == 0:
            message = tf.zeros_like(message_policy)
        else:
            message = (
                message_policy.sample()
                if isinstance(message_policy, tfd.Distribution)
                else message_policy
            )
        return message
