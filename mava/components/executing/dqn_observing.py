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
import copy
from types import SimpleNamespace
from typing import Any, Dict, List, Type

from mava.callbacks import Callback
from mava.components import Component
from mava.components.building.adders import Adder
from mava.components.building.parameter_client import ExecutorParameterClient
from mava.components.building.system_init import BaseSystemInit
from mava.components.executing.action_selection import ExecutorSelectAction
from mava.core_jax import SystemExecutor
from mava.systems.idqn.components.executing.observing import FeedforwardExecutorObserve
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num


class IDRQNExecutorObserve(FeedforwardExecutorObserve):
    def __init__(self, config: SimpleNamespace = SimpleNamespace()):
        """Component handles observations for a feedforward executor.

        Args:
            config: SimpleNamespace.
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

        # Initialise the recurrent states of the agents
        executor.store.policy_states = {}
        for agent in executor.store.agent_net_keys.keys():
            network = executor.store.agent_net_keys[agent]
            executor.store.policy_states[agent] = copy.copy(
                executor.store.networks[network].get_init_state()
            )

        # Return if the executor has no adder.
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

        executor.store.extras["policy_states"] = executor.store.policy_states

        # executor.store.timestep set by Executor
        #TODO: USE INHERITANCE
        executor.store.adder.add_first(executor.store.timestep)#, executor.store.extras)

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

        adder_actions: Dict[str, Any] = {}
        # executor.store.next_extras set by Executor
        for agent in actions_info.keys():
            adder_actions[agent] = {
                "actions_info": actions_info[agent],
            }

        executor.store.extras[
            "network_int_keys"
        ] = executor.store.network_int_keys_extras

        # executor.store.extras set by Executor
        executor.store.extras["policy_states"] = executor.store.policy_states

        executor.store.adder.add(
            adder_actions, executor.store.next_timestep, executor.store.extras
        )
