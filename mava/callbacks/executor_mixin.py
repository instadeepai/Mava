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

    def on_execution_init_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_init_start(self)

    def on_execution_init(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_init(self)

    def on_execution_init_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_init_end(self)

    def on_execution_select_action_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_action_start(self)

    def on_execution_select_action_preprocess(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_action_preprocess(self)

    def on_execution_select_action_compute(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_action_compute(self)

    def on_execution_select_action_sample(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_action_sample(self)

    def on_execution_select_action_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_action_end(self)

    def on_execution_observe_first_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_observe_first_start(self)

    def on_execution_observe_first(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_observe_first(self)

    def on_execution_observe_first_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_observe_first_end(self)

    def on_execution_observe_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_observe_start(self)

    def on_execution_observe(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_observe(self)

    def on_execution_observe_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_observe_end(self)

    def on_execution_select_actions_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_actions_start(self)

    def on_execution_select_actions(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_actions(self)

    def on_execution_select_actions_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_select_actions_end(self)

    def on_execution_update_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_update_start(self)

    def on_execution_update(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_update(self)

    def on_execution_update_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_execution_update_end(self)
