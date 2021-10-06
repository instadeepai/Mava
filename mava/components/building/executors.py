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

"""Execution components for system builders"""

from typing import List, Dict, Type

from acme.specs import EnvironmentSpec

from mava import core
from mava.callbacks import Callback
from mava.systems.building import SystemBuilder


class Executor(Callback):
    def __init__(
        self,
        executor_fn: Type[core.Executor],
        net_to_ints: Dict[str, int],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        executor_samples: List[str],
    ):
        """[summary]

        Args:
            net_to_ints (Dict[str, int]): [description]
            agent_specs (Dict[str, EnvironmentSpec]): [description]
            agent_net_keys (Dict[str, str]): [description]
            executor_samples (executor_samples): [description]
        """
        self.executor_fn = executor_fn
        self.net_to_ints = net_to_ints
        self.agent_specs = agent_specs
        self.agent_net_keys = agent_net_keys
        self.executor_samples = executor_samples

    def on_building_executor(self, builder: SystemBuilder):
        """[summary]

        Args:
            builder (SystemBuilder): [description]
        """
        builder.executor = self.executor_fn(
            policy_networks=self._policy_networks,
            counts=builder.counts,
            net_to_ints=self.net_to_ints,
            agent_specs=self.agent_specs,
            agent_net_keys=self.agent_net_keys,
            executor_samples=self.executor_samples,
            variable_client=builder.variable_client,
            adder=self._adder,
        )