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
from typing import Any, Dict, Callable, Optional
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

    # Update the executor variables.networks']['network_agent
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
        #print(dir(executor.store))
        #if dir(executor.store.extras_finder):
        #TODO: ADD PROPER IF STATEMENT FOR TRANSITION AND TRAJECTORY ADDER USAGE
        if 1==1:
            keys = list(executor.store.next_extras_specs.keys())
            extras = executor.store.extras_finder(executor.store, keys)
            executor.store.extras.update(extras)

            executor.store.adder.add_first(executor.store.timestep, executor.store.extras)
            
            executor.store.keys_available_as_next_extra = list(extras.keys())
        
        else:
           executor.store.adder.add_first(executor.store.timestep, executor.store.extras)
       
        

    # Observe
    def on_execution_observe(self, executor: SystemExecutor) -> None:
        """_summary_print(key)
                print(value)

        Args:
            executor : _description_
        """
        if not executor.store.adder:
            return

        if 1==1:
            #print("OBSERVE")
            # collecting user-defined extras
            keys = list(executor.store.next_extras_specs.keys())
            extras = executor.store.extras_finder(executor.store, keys)
            #print(extras)
            #exit()
            executor.store.next_extras.update(extras)
            
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
            all_keys = list(executor.store.extras_specs.keys())
            keys_to_be_removed = list(executor.store.next_extras_specs.keys())  # they are
            # already there.
            for key in keys_to_be_removed:
                if key in all_keys:
                    all_keys.remove(key)
            keys = all_keys
            extras = executor.store.extras_finder(executor.store, keys)

            policy_info = {'policy_info': 
                    {'agent_0': {'action_values': jnp.array([0.,0.,0.,0.,0.])}
                    ,'agent_1': {'action_values': jnp.array([0.,0.,0.,0.,0.])}
                    ,'agent_2': {'action_values': jnp.array([0.,0.,0.,0.,0.])}}}


            #print(extras)
            #print(executor.store.extras)
            #exit()
            #print(executor.store.next_extras)
            #exit()
            executor.store.adder.add(
                actions=adder_actions,
                next_timestep=executor.store.next_timestep,
                next_extras=executor.store.next_extras,
                #extras=extras,
            )
        else:
            
            #FOR MAPPO
            actions_info = executor.store.actions_info
            policies_info = executor.store.policies_info


            #GET EXTRAS FOR DQN

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
            
            #print(executor.store.next_extras)
            #exit()
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
