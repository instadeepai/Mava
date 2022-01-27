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

"""System executor setup implementation."""
from typing import Any, Dict, List, Optional

import numpy as np
import sonnet as snt
from acme.specs import EnvironmentSpec

from mava import adders
from mava.callbacks import Callback
from mava.core import SystemExecutor
from mava.systems import VariableClient


class ExecutorSetup(Callback):
    """Setup for executor"""

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        network_sampling_setup: List,
        net_keys_to_ids: Dict[str, int],
        evaluator: bool = False,
        adder: Optional[adders.ReverbParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[VariableClient] = None,
        interval: Optional[dict] = None,
    ):

        """Initialise the system executor"""

        # Store these for later use.
        self._policy_networks = policy_networks
        self._agent_net_keys = agent_net_keys
        self._adder = adder
        self._variable_client = variable_client
        self._agent_specs = agent_specs
        self._network_sampling_setup = network_sampling_setup
        self._counts = counts
        self._network_int_keys_extras: Dict[str, np.ndarray] = {}
        self._net_keys_to_ids = net_keys_to_ids
        self._evaluator = evaluator
        self._interval = interval

    def on_execution_init(self, executor: SystemExecutor) -> None:
        executor.policy_networks = self._policy_networks
        executor.agent_net_keys = self._agent_net_keys
        executor.adder = self._adder
        executor.variable_client = self._variable_client
        executor.agent_specs = self._agent_specs
        executor.network_sampling_setup = self._network_sampling_setup
        executor.counts = self._counts
        executor.network_int_keys_extras = self._network_int_keys_extras
        executor.net_keys_to_ids = self._net_keys_to_ids
        executor.evaluator = self._evaluator
        executor.interval = self._interval
