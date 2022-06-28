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


class TrainerHookMixin(ABC):

    ######################
    # system trainer hooks
    ######################

    callbacks: List

    def on_training_init_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init_start(self)

    def on_training_init(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init(self)

    def on_training_init_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_init_end(self)

    def on_training_utility_fns(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_utility_fns(self)

    def on_training_loss_fns(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_loss_fns(self)

    def on_training_step_fn(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_step_fn(self)

    def on_training_step_start(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_step_start(self)

    def on_training_step(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_step(self)

    def on_training_step_end(self) -> None:
        """[summary]"""
        for callback in self.callbacks:
            callback.on_training_step_end(self)
