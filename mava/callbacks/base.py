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

"""
Abstract base class used to build new callbacks.
"""

import abc
from typing import Any, Dict, List, Optional, Type

from mava.systems.system import System
from mava.systems.building import SystemBuilder


class Callback(abc.ABC):
    """
    Abstract base class used to build new callbacks.
    Subclass this class and override any of the relevant hooks
    """

    ######################
    # system builder hooks
    ######################

    def on_building_init_start(self, system: System, builder: SystemBuilder) -> None:
        """[summary]

        Args:
            system (System): [description]
            builder (SystemBuilder): [description]
        """
        pass

    def on_building_init_end(self, system: System, builder: SystemBuilder) -> None:
        """Called when the builder initialisation ends."""
        pass

    def on_building_make_replay_table_start(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[description]"""
        pass

    def on_building_adder_signature(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_rate_limiter(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_tables(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_replay_table_end(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_make_dataset_iterator_start(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_dataset(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_dataset_iterator_end(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_make_adder_start(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_adder_priority(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_make_adder(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_tables(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_replay_table_start(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_make_variable_server_start(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_variable_server(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_make_variable_server_end(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_make_executor_start(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_executor_variable_client(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_executor(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_executor_end(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_make_trainer_start(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_trainer_variable_client(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_trainer(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_statistics(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_make_trainer_end(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_execution_init_start(self, executor: SystemExecutor) -> None:
        """[summary]

        Args:
            executor (SystemExecutor): [description]
        """
        pass

    def on_execution_init_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_policy_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_policy_preprocess(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_policy_compute(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_policy_sample_action(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_policy_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_action_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_action(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_action_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe_first_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe_first(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe_first_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_actions_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_actions_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_update_start(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_update(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_update_end(self, executor: SystemExecutor) -> None:
        """[summary]"""
        pass