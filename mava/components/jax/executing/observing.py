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
from typing import Any, Callable, Dict, Optional

from mava.components.jax import Component
from mava.core_jax import SystemExecutor
from mava.utils.extras.extras import UserDefinedExtrasFinder
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num

# @dataclass
# class ExtrasConfig:
#     extra_creator_s: ExtrasSyncWithStateCreator = ExtrasSyncWithStateCreator()
#     extra_creator_a: ExtrasSyncWithActionCreator = ExtrasSyncWithActionCreator()
#
#
# class Extras(Component):
#     def __init__(self, config: ExtrasConfig = ExtrasConfig()):
#         """Creating Extras from Store of the executor at its current state."""
#         self.config = config
#
#     def on_execution_init(self, executor: SystemExecutor) -> None:
#         executor.store.extras_sync_with_state = self.config.extra_creator_s
#         executor.store.extras_sync_with_action = self.config.extra_creator_a
#
#     @staticmethod
#     def name() -> str:
#         """_summary_"""
#         return "extras"
#
#     @staticmethod
#     def config_class() -> Optional[Callable]:
#         """
#         Optional class which specifies the
#         dataclass/config object for the component.
#         """
#         return ExtrasConfig
#


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

        # At this point executor.store.extras should only have env_extras.
        # In the following we get the user defined extras which are synced with state
        # and update the executor.store.extra dictionary with them

        # collecting user-defined extras
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

    # Observe
    def on_execution_observe(self, executor: SystemExecutor) -> None:
        """_summary_

        Args:
            executor : _description_
        """
        if not executor.store.adder:
            return

        # At this point the state is t+1, and next_extra has the env_extra at t+1.
        # In the following, we get the user-defined extras which are synced with the
        # state and update the executor.store.next_extra dictionary with them. By
        # calling the adder, this information is going to (partially) appended to the
        # reverb trajectory under:
        # --> item "extra"
        # --> at time step t+1; hence it is called next_extra.
        # tmp = executor.store.extras_sync_with_state.create(executor.store)
        keys = list(executor.store.next_extras_spec.keys())
        extras = executor.store.extras_finder(executor.store, keys)
        executor.store.next_extras.update(extras)

        # We are not yet done with the timestep t, the associated actions and the extra
        # information which is  synced with its action(i.e. the latest action) are
        # still not stored. An example of extra information which is synced with the
        # action is action-values or policy values. This information unlike the
        # information created above will be appended to the reverb trajectory under
        # timestep t (and not timestep t+1). That's why they are extras (and not next
        # extras). A more expressive name would be extras_synced_with_actions,
        # but we stick to extras.

        actions_info = executor.store.actions_info  # includes taken actions
        adder_actions: Dict[str, Any] = {}

        for agent in actions_info.keys():
            adder_actions[agent] = {
                "actions_info": actions_info[agent],
            }

        # the extra information which became in relation to the taken actions (
        # actions of timestep t). As these are information which are related to the
        # timestep t (and not timestep t+1), they belong to the extras.
        # extras = executor.store.extras_sync_with_action.create(executor.store)
        all_keys = list(executor.store.extras_spec.keys())
        keys_to_be_removed = list(executor.store.next_extras_spec.keys())  # they are
        # already there.
        for key in keys_to_be_removed:
            if key in all_keys:
                all_keys.remove(key)
        keys = all_keys
        extras = executor.store.extras_finder(executor.store, keys)

        executor.store.adder.add(
            actions=adder_actions,
            next_timestep=executor.store.next_timestep,
            next_extras=executor.store.next_extras,
            extras=extras,
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
