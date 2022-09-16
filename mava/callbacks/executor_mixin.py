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

"""Abstract mixin class used to call system component hooks."""

from abc import ABC
from typing import List


class ExecutorHookMixin(ABC):

    #######################
    # system executor hooks
    #######################

    callbacks: List

    # INIT
    def on_execution_init_start(self) -> None:
        """Start of executor initialisation."""
        for callback in self.callbacks:
            callback.on_execution_init_start(self)

    def on_execution_init(self) -> None:
        """Executor initialisation."""
        for callback in self.callbacks:
            callback.on_execution_init(self)

    def on_execution_init_end(self) -> None:
        """End of executor initialisation."""
        for callback in self.callbacks:
            callback.on_execution_init_end(self)

    # SELECT ACTION
    def on_execution_select_action_start(self) -> None:
        """Start of executor selecting an action for agent."""
        for callback in self.callbacks:
            callback.on_execution_select_action_start(self)

    def on_execution_select_action_preprocess(self) -> None:
        """Preprocessing when executor selecting an action for agent."""
        for callback in self.callbacks:
            callback.on_execution_select_action_preprocess(self)

    def on_execution_select_action_compute(self) -> None:
        """Call to agent networks when executor selecting an action for agent."""
        for callback in self.callbacks:
            callback.on_execution_select_action_compute(self)

    def on_execution_select_action_sample(self) -> None:
        """Sample an action when executor selecting an action for agent."""
        for callback in self.callbacks:
            callback.on_execution_select_action_sample(self)

    def on_execution_select_action_end(self) -> None:
        """End of executor selecting an action for agent."""
        for callback in self.callbacks:
            callback.on_execution_select_action_end(self)

    # OBSERVE FIRST
    def on_execution_observe_first_start(self) -> None:
        """Start of executor observing the first time in an episode."""
        for callback in self.callbacks:
            callback.on_execution_observe_first_start(self)

    def on_execution_observe_first(self) -> None:
        """Executor observing the first time in an episode."""
        for callback in self.callbacks:
            callback.on_execution_observe_first(self)

    def on_execution_observe_first_end(self) -> None:
        """End of executor observing the first time in an episode."""
        for callback in self.callbacks:
            callback.on_execution_observe_first_end(self)

    # OBSERVE
    def on_execution_observe_start(self) -> None:
        """Start of executor observing."""
        for callback in self.callbacks:
            callback.on_execution_observe_start(self)

    def on_execution_observe(self) -> None:
        """Executor observing."""
        for callback in self.callbacks:
            callback.on_execution_observe(self)

    def on_execution_observe_end(self) -> None:
        """End of executor observing."""
        for callback in self.callbacks:
            callback.on_execution_observe_end(self)

    # SELECT ACTIONS
    def on_execution_select_actions_start(self) -> None:
        """Start of executor selecting actions for all agents in the system."""
        for callback in self.callbacks:
            callback.on_execution_select_actions_start(self)

    def on_execution_select_actions(self) -> None:
        """Executor selecting actions for all agents in the system."""
        for callback in self.callbacks:
            callback.on_execution_select_actions(self)

    def on_execution_select_actions_end(self) -> None:
        """End of executor selecting actions for all agents in the system."""
        for callback in self.callbacks:
            callback.on_execution_select_actions_end(self)

    # UPDATE
    def on_execution_update_start(self) -> None:
        """Start of updating executor parameters."""
        for callback in self.callbacks:
            callback.on_execution_update_start(self)

    def on_execution_update(self) -> None:
        """Update executor parameters."""
        for callback in self.callbacks:
            callback.on_execution_update(self)

    def on_execution_update_end(self) -> None:
        """End of updating executor parameters."""
        for callback in self.callbacks:
            callback.on_execution_update_end(self)

    # FORCE UPDATE
    def on_execution_force_update_start(self) -> None:
        """Start of forcing the update of the executor parameters."""
        for callback in self.callbacks:
            callback.on_execution_force_update_start(self)

    def on_execution_force_update(self) -> None:
        """Froce update executor parameters."""
        for callback in self.callbacks:
            callback.on_execution_force_update(self)

    def on_execution_force_update_end(self) -> None:
        """End of forcing the update of the executor parameters."""
        for callback in self.callbacks:
            callback.on_execution_force_update_end(self)
