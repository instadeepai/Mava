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

from dataclasses import dataclass
from typing import Any, Dict

from mava.components.jax import Component
from mava.core_jax import SystemExecutor
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num


@dataclass
class ExecutorObserveConfig:
    pass


class FeedforwardExecutorObserve(Component):
    def __init__(self, config: ExecutorObserveConfig = ExecutorObserveConfig()):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    # Observe first
    def on_execution_observe_first(self, executor: SystemExecutor) -> None:
        """_summary_

        Args:
            executor : _description_
        """
        if not executor.store.adder:
            return

        "Select new networks from the sampler at the start of each episode."
        agents = sort_str_num(list(executor.store.agent_net_keys.keys()))
        (
            executor.store.network_int_keys_extras,
            executor.store.agent_net_keys,
        ) = sample_new_agent_keys(
            agents,
            executor.store.network_sampling_setup,
            executor.store.net_keys_to_ids,
        )
        executor.store.extras[
            "network_int_keys"
        ] = executor.store.network_int_keys_extras

        executor.store.adder.add_first(executor.store.timestep, executor.store.extras)

    # Observe
    def on_execution_observe(self, executor: SystemExecutor) -> None:
        """_summary_

        Args:
            executor : _description_
        """
        if not executor.store.adder:
            return

        actions_info = executor.store.actions_info
        policies_info = executor.store.policies_info

        adder_actions: Dict[str, Any] = {}
        executor.store.next_extras["policy_info"] = {}
        for agent in actions_info.keys():
            adder_actions[agent] = {
                "actions_info": actions_info[agent],
            }
            executor.store.next_extras["policy_info"][agent] = policies_info[agent]

        executor.store.next_extras[
            "network_int_keys"
        ] = executor.store.network_int_keys_extras

        executor.store.adder.add(
            adder_actions, executor.store.next_timestep, executor.store.next_extras
        )

    # Update the executor variables.
    def on_execution_update(self, executor: SystemExecutor) -> None:
        """Update the policy variables."""
        if executor.store.executor_parameter_client:
            executor.store.executor_parameter_client.get_async()

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "executor_observe"
