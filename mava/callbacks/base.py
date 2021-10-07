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

from mava.systems.system import System
from mava.systems.building import SystemBuilder
from mava.systems.execution import SystemExecutor


class Callback(abc.ABC):
    """
    Abstract base class used to build new callbacks.
    Subclass this class and override any of the relevant hooks
    """

    ######################
    # system builder hooks
    ######################

    # initialisation
    def on_building_init_start(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_init(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_init_end(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # tables
    def on_building_tables_start(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_adder_signature(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_rate_limiter(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_tables(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_tables_end(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # dataset
    def on_building_dataset_iterator_start(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_dataset(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_dataset_iterator_end(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    # adder
    def on_building_adder_start(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_adder_priority(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_adder(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_adder_end(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # variable server
    def on_building_variable_server_start(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_variable_server(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_variable_server_end(
        self, system: System, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    # executor
    def on_building_executor_start(
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

    def on_building_executor_end(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # trainer
    def on_building_trainer_start(self, system: System, builder: SystemBuilder) -> None:
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

    def on_building_trainer_end(self, system: System, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    ######################
    # system executor hooks
    ######################

    def on_execution_init_start(self, system: System, executor: SystemExecutor) -> None:
        """[summary]

        Args:
            executor (SystemExecutor): [description]
        """
        pass

    def on_execution_init_end(self, system: System, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_policy_start(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_policy_preprocess(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_policy_compute(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_policy_sample_action(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_policy_end(self, system: System, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_select_action_start(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_select_action(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_select_action_end(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_observe_first_start(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_observe_first(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_observe_first_end(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_observe_start(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_observe(self, system: System, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_observe_end(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_select_actions_start(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_select_actions(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_select_actions_end(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_update_start(
        self, system: System, executor: SystemExecutor
    ) -> None:
        """[summary]"""
        pass

    def on_execution_update(self, system: System, executor: SystemExecutor) -> None:
        """[summary]"""
        pass

    def on_execution_update_end(self, system: System, executor: SystemExecutor) -> None:
        """[summary]"""
        pass