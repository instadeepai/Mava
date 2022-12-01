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

"""DQN Observation components for system builders"""

from types import SimpleNamespace
from typing import Any, Dict

from mava.components.executing.observing import FeedforwardExecutorObserve
from mava.core_jax import SystemExecutor


class DQNFeedforwardExecutorObserve(FeedforwardExecutorObserve):
    def __init__(self, config: SimpleNamespace = SimpleNamespace()):
        """Component handles observations for a feedforward executor.

        Args:
            config: SimpleNamespace.
        """
        self.config = config

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

        network_int_keys = executor.store.network_int_keys_extras
        executor.store.next_extras["network_int_keys"] = network_int_keys

        # executor.store.next_timestep set by Executor
        executor.store.adder.add(
            adder_actions, executor.store.next_timestep, executor.store.next_extras
        )
