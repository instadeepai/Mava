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
from typing import Dict, Optional

import sonnet as snt
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.callbacks import Callback
from mava.systems.training import SystemTrainer


class Trainer(Callback):
    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ) -> None:

        # Store these for later use.
        self._policy_networks = policy_networks
        self._agent_net_keys = agent_net_keys
        self._adder = adder
        self._variable_client = variable_client

    def on_training_init_start(self, trainer: SystemTrainer) -> None:

        executor._policy_networks = self._policy_networks
        executor._agent_net_keys = self._agent_net_keys
        executor._adder = self._adder
        executor._variable_client = self._variable_client