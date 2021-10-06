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

"""Commonly used adder components for system builders"""
from typing import Dict, Optional, List, Any

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from dm_env import specs
from acme.specs import EnvironmentSpec
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.callbacks import Callback
from mava.systems.execution import SystemExecutor

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray
tfd = tfp.distributions


class MADDPGFeedForwardExecutor(Callback):
    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        executor_samples: List,
        net_to_ints: Dict[str, int],
        adder: Optional[adders.ParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):

        """Initialise the system executor
        Args:
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            agent_specs (Dict[str, EnvironmentSpec]): agent observation and action
                space specifications.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
        """

        # Store these for later use.
        self._policy_networks = policy_networks
        self._agent_specs = agent_specs
        self._executor_samples = executor_samples
        self._agent_net_keys = agent_net_keys
        self._counts = counts
        self._network_int_keys_extras: Dict[str, np.array] = {}
        self._net_to_ints = net_to_ints
        self._adder = adder
        self._variable_client = variable_client

    def on_execution_init_start(self, executor: SystemExecutor) -> None:
        """[summary]

        Args:
            executor (SystemExecutor): [description]
        """
        executor._policy_networks = self._policy_networks
        executor._agent_specs = self._agent_specs
        executor._executor_samples = self._executor_samples
        executor._agent_net_keys = self._agent_net_keys
        executor._counts = self._counts
        executor._network_int_keys_extras = self._network_int_keys_extras
        executor._net_to_ints = self._net_to_ints
        executor._adder = self._adder
        executor._variable_client = self._variable_client

    def on_execution_policy_sample_action(self, executor: SystemExecutor) -> None:
        if type(self._agent_specs[executor._agent].actions) == BoundedArray:
            # Continuous action
            action = executor.policy
        elif type(self._agent_specs[executor._agent].actions) == DiscreteArray:
            action = tf.math.argmax(executor.policy, axis=1)
        else:
            raise NotImplementedError

        executor.action_info = (action, executor.policy)