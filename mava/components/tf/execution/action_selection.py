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

from acme.tf import utils as tf2_utils

from mava.core import SystemExecutor
from mava.components.execution import ActionSelector


class OnlineActionSampling(ActionSelector):
    def on_execution_policy_sample_action(self, executor: SystemExecutor) -> None:

        # Sample from the policy if it is stochastic.
        action = executor.policy.sample()

        executor.action_info = action

    def on_execution_select_action(self, executor: SystemExecutor) -> None:

        # Pass the observation through the policy network.
        action = self._policy(self._agent, self._observation.observation)

        executor.action = action

    def on_execution_select_action_end(self, executor: SystemExecutor) -> None:
        executor.action = tf2_utils.to_numpy_squeeze(executor.action)

    def on_execution_select_actions(self, executor: SystemExecutor) -> None:

        actions = {}
        for agent, observation in self._observations.items():
            action = self._policy(agent, observation.observation)
            actions[agent] = tf2_utils.to_numpy_squeeze(action)

        executor.actions = actions
