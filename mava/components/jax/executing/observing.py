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

import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import jax.numpy as jnp

from mava.components.jax import Component
from mava.core_jax import SystemExecutor
from mava.utils.extras.extras import UserDefinedExtrasFinder
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num


@dataclass
class ExtrasFinderConfig:
    extras_finder: UserDefinedExtrasFinder = UserDefinedExtrasFinder()


class ExtrasFinder(Component):
    def __init__(self, config: ExtrasFinderConfig = ExtrasFinderConfig()):
        """Creating Extras from Store of the executor at its current state."""
        self.config = config

    def on_execution_init(self, executor: SystemExecutor) -> None:
        """The function for finding extras are added to store."""
        executor.store.extras_finder = self.config.extras_finder.find

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "extras_finder"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Returns class config."""
        return ExtrasFinderConfig


@dataclass
class ExecutorObserveConfig:
    pass


class ExecutorObserve(Component):
    @abc.abstractmethod
    def __init__(self, config: ExecutorObserveConfig = ExecutorObserveConfig()):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    # Observe first
    @abc.abstractmethod
    def on_execution_observe_first(self, executor: SystemExecutor) -> None:
        """_summary_

        Args:
            executor : _description_
        """
        pass

    # Observe
    @abc.abstractmethod
    def on_execution_observe(self, executor: SystemExecutor) -> None:
        """_summary_

        Args:
            executor : _description_
        """
        pass

    # Update the executor variables.
    @abc.abstractmethod
    def on_execution_update(self, executor: SystemExecutor) -> None:
        """Update the policy variables."""
        pass

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "executor_observe"


class FeedforwardExecutorObserve(ExecutorObserve):
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

        # keys = list(executor.store.extras_spec.keys())
        # extras = executor.store.extras_finder(executor.store, keys)
        # executor.store.next_extras.update(extras)

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

        # fake data
        policy_info = {
            "policy_info": {
                "agent_0": {"action_values": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])},
                "agent_1": {"action_values": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])},
                "agent_2": {"action_values": jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])},
            }
        }
        executor.store.extras[
            "network_int_keys"
        ] = executor.store.network_int_keys_extras

        """
        executor.store.extras[
            "policy_info"
        ] = policy_info
        """

        # print(executor.store.extras)
        # actions_info = executor.store.actions_info
        # policies_info = executor.store.policies_info
        # exit()
        print(executor.store.extras)
        # exit()
        executor.store.adder.add_first(executor.store.timestep, executor.store.extras)

        # executor.store.keys_available_as_next_extra = list(extras.keys())
        # print(executor.store.extras)
        # exit()
        # TODO: ADD CORRECT UPDATED STORE!!!!!
        # executor.store.adder.add_first(executor.store.timestep, executor.store.extras)
        # actions_info = executor.store.actions_info
        # policies_info = executor.store.policies_info

        # print(policies_info)
        # print(actions_info)
        # collecting user-defined extras
        # print(executor.store)
        # exit()
        """
        keys = list(executor.store.next_extras_spec.keys())
        extras = executor.store.extras_finder(executor.store, keys)
        executor.store.extras.update(extras)
        # Now we add all the extras which are synced with the state (env_extras plus
        # user defined ones).
        executor.store.adder.add_first(executor.store.timestep, executor.store.extras)

        # The following variable in executor.store keeps track of which elements of the
        # "extra"
        # are available before taking the decision of the current step. These are the
        # information which are known right after taking the action of the previous
        # step.
        executor.store.keys_available_as_next_extra = list(extras.keys())
        """

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

        # GET EXTRAS FOR DQN

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

        # print(executor.store.extras)
        print(executor.store.next_extras)
        exit()
        executor.store.adder.add(
            adder_actions, executor.store.next_timestep, executor.store.next_extras
        )

    # Update the executor variables.
    def on_execution_update(self, executor: SystemExecutor) -> None:
        """Update the policy variables."""
        if executor.store.executor_parameter_client:
            executor.store.executor_parameter_client.get_async()
