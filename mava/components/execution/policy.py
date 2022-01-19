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

import abc

from mava.callbacks import Callback
from mava.core import SystemExecutor


class Policy(Callback):
    @abc.abstractmethod
    def on_execution_policy_compute(self, executor: SystemExecutor) -> None:
        """[summary]

        Args:
            executor (SystemExecutor): [description]
        """


class DistributionPolicy(Policy):
    def on_execution_policy_compute(self, executor: SystemExecutor) -> None:
        # index network either on agent type or on agent id
        agent_key = self._agent_net_keys[self._agent]

        # Compute the policy, conditioned on the observation.
        policy = self._policy_networks[agent_key](executor.processed_observation)

        executor.policy = policy
