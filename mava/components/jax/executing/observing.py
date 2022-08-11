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

"""Observation components for system builders"""

import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Type

from mava.callbacks import Callback
from mava.components.jax import Component
from mava.components.jax.building.adders import Adder
from mava.components.jax.building.parameter_client import ExecutorParameterClient
from mava.components.jax.building.system_init import BaseSystemInit
from mava.components.jax.executing.action_selection import ExecutorSelectAction
from mava.core_jax import SystemExecutor
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num


@dataclass
class ExecutorObserveConfig:
    pass


class ExecutorObserve(Component):
    @abc.abstractmethod
    def __init__(self, config: ExecutorObserveConfig = ExecutorObserveConfig()):
        """Abstract component parses observations and updates executor variables.

        Args:
            config: ExecutorObserveConfig.
        """
        self.config = config

    @abc.abstractmethod
    def on_execution_observe_first(self, executor: SystemExecutor) -> None:
        """Handle first executor observation in episode."""
        pass

    @abc.abstractmethod
    def on_execution_observe(self, executor: SystemExecutor) -> None:
        """Handle observations in executor."""
        pass

    @abc.abstractmethod
    def on_execution_update(self, executor: SystemExecutor) -> None:
        """Update the executor variables."""
        pass

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "executor_observe"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        Adder required to set up executor.store.adder.
        BaseSystemInit required to set up executor.store.agent_net_keys,
        executor.store.network_sampling_setup, and executor.store.net_keys_to_ids.
        ExecutorSelectAction required to assign executor.store.actions_info
        and executor.store.policies_info.
        ExecutorParameterClient required to set up
        executor.store.executor_parameter_client.

        Returns:
            List of required component classes.
        """
        return [Adder, BaseSystemInit, ExecutorSelectAction, ExecutorParameterClient]


class FeedforwardExecutorObserve(ExecutorObserve):
    def __init__(self, config: ExecutorObserveConfig = ExecutorObserveConfig()):
        """Component handles observations for a feedforward executor.

        Args:
            config: ExecutorObserveConfig.
        """
        self.config = config

    def on_execution_observe_first(self, executor: SystemExecutor) -> None:
        """Handle first observation in episode and give to adder.

        Also selects networks to be used for episode.

        Args:
            executor: SystemExecutor.

        Returns:
            None.
        """
        if not executor.store.adder:
            return

        # Select new networks from the sampler at the start of each episode.
        agents = sort_str_num(list(executor.store.agent_net_keys.keys()))
        (
            executor.store.network_int_keys_extras,
            executor.store.agent_net_keys,
        ) = sample_new_agent_keys(
            agents,
            executor.store.network_sampling_setup,
            executor.store.net_keys_to_ids,
        )
        # executor.store.extras set by Executor
        executor.store.extras[
            "network_int_keys"
        ] = executor.store.network_int_keys_extras

        # executor.store.timestep set by Executor
        executor.store.adder.add_first(executor.store.timestep, executor.store.extras)

    def on_execution_observe(self, executor: SystemExecutor) -> None:
        """Handle observations and pass along to the adder.

        Args:
            executor: SystemExecutor.

        Returns:
            None.
        """
        if not executor.store.adder:
            return

        actions_info = executor.store.actions_info
        policies_info = executor.store.policies_info

        adder_actions: Dict[str, Any] = {}
        # executor.store.next_extras set by Executor
        executor.store.next_extras["policy_info"] = {}
        for agent in actions_info.keys():
            adder_actions[agent] = {
                "actions_info": actions_info[agent],
            }
            executor.store.next_extras["policy_info"][agent] = policies_info[agent]

        executor.store.next_extras[
            "network_int_keys"
        ] = executor.store.network_int_keys_extras

        # executor.store.next_timestep set by Executor
        executor.store.adder.add(
            adder_actions, executor.store.next_timestep, executor.store.next_extras
        )

    def on_execution_update(self, executor: SystemExecutor) -> None:
        """Update the executor variables."""
        if executor.store.executor_parameter_client:
            executor.store.executor_parameter_client.get_async()
