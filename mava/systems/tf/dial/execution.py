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
from typing import Any, Dict, Optional, Tuple

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.systems.tf.executors import RecurrentExecutor

tfd = tfp.distributions


class DIALExecutor(RecurrentExecutor):
    """DIAL implementation of a recurrent Executor."""

    def __init__(
        self,
        policy_networks: Dict[str, snt.RNNCore],
        communication_module: BaseCommunicationModule,
        message_size: int = 1,
        shared_weights: bool = True,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
    ):
        """Initializes the executor.
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
        self._shared_weights = shared_weights
        self._adder = adder
        self._variable_client = variable_client
        self._policy_networks = policy_networks
        self._message_size = message_size
        self._states: Dict[str, Any] = {}
        self._messages: Dict[str, Any] = {}
        self._prev_states: Dict[str, Any] = {}
        self._prev_messages: Dict[str, Any] = {}
        self._store_recurrent_state = store_recurrent_state
        self._communication_module = communication_module

    @tf.function
    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        state: types.NestedTensor,
        message: types.NestedTensor,
    ) -> Tuple[types.NestedTensor, types.NestedTensor, types.NestedTensor]:

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # index network either on agent type or on agent id
        agent_key = agent.split("_")[0] if self._shared_weights else agent

        # Compute the policy, conditioned on the observation.
        (action_policy, message_policy), new_state = self._policy_networks[agent_key](
            batched_observation, state, message
        )

        # action_policy = policy[:,:-self._message_size]
        # message_policy = policy[:,self._message_size:]

        # Sample from the policy if it is stochastic.
        action = (
            action_policy.sample()
            if isinstance(action_policy, tfd.Distribution)
            else action_policy
        )
        message = (
            message_policy.sample()
            if isinstance(message_policy, tfd.Distribution)
            else message_policy
        )

        return action, message, new_state

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Optional[Dict[str, types.NestedArray]] = {},
    ) -> None:
        super().observe_first(timestep=timestep, extras=extras)

        # Re-initialize the RNN state.
        for agent, _ in timestep.observation.items():
            # index network either on agent type or on agent id
            # agent_key = agent.split("_")[0] if self._shared_weights else agent
            self._messages[agent] = tf.zeros(self._message_size, dtype=tf.float32)

    def select_action(
        self,
        agent: str,
        observation: types.NestedArray,
    ) -> Tuple[types.NestedArray, types.NestedArray]:
        """Get actions and messages of specific agent"""

        # Initialize the RNN state if necessary.
        if self._states[agent] is None:
            self._states[agent] = self._networks[agent].initial_state(1)

        # Step the recurrent policy forward given the current observation and state.
        policy_output, new_state = self._policy(
            agent, observation.observation, self._states[agent]
        )

        # Bookkeeping of recurrent states for the observe method.
        self._update_state(agent, new_state)

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(policy_output).argmax()

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Dict[str, types.NestedArray]:
        actions = {}

        message_inputs = self._communication_module.process_messages(self._messages)

        for agent, observation in observations.items():

            # Step the recurrent policy forward given the current observation and state.
            policy_output, message, new_state = self._policy(
                agent,
                observation.observation,
                self._states[agent],
                message_inputs[agent],
            )

            # Bookkeeping of recurrent states for the observe method.
            self._states[agent] = new_state
            self._messages[agent] = message

            # self._update_state(agent, new_state)
            # TODO Mask actions here using observation.legal_actions
            # What happens in discrete vs cont case
            actions[agent] = tf2_utils.to_numpy_squeeze(policy_output)

        # Return a numpy array with squeezed out batch dimension.
        return actions
