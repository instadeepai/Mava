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

import sonnet as snt
from acme import types
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.systems.tf.executors import RecurrentExecutor


class DIALExecutor(RecurrentExecutor):
    """DIAL implementation of a recurrent Executor."""

    def __init__(
        self,
        policy_networks: Dict[str, snt.RNNCore],
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
        self._adder = adder
        self._variable_client = variable_client
        self._networks = policy_networks
        self._states: Dict[str, Any] = {}
        self._prev_states: Dict[str, Any] = {}
        self._store_recurrent_state = store_recurrent_state

    def select_action_message(
        self, agent: str, observation: types.NestedArray
    ) -> Tuple[types.NestedArray, types.NestedArray]:
        """Get actions and messages of specific agent"""

    def select_actions_messages(
        self, observations: Dict[str, types.NestedArray]
    ) -> Tuple[types.NestedArray, types.NestedArray]:
        """Get actions and messages of all agents"""
