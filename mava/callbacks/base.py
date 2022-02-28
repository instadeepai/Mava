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

"""Abstract base class used to build new system components."""

import abc

from mava.core_jax import SystemBuilder


class Callback(abc.ABC):
    """Abstract base class used to build new components. \
        Subclass this class and override any of the relevant hooks."""

    ######################
    # system builder hooks
    ######################

    # BUILDER INITIAlISATION
    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """Start of builder initialisation."""
        pass

    def on_building_init(self, builder: SystemBuilder) -> None:
        """Builder initialisation."""
        pass

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """End of builder initialisation."""
        pass

    # DATA SERVER
    def on_building_tables_start(self, builder: SystemBuilder) -> None:
        """Start of data server table building."""
        pass

    def on_building_tables_adder_signature(self, builder: SystemBuilder) -> None:
        """Building of table adder signature."""
        pass

    def on_building_tables_rate_limiter(self, builder: SystemBuilder) -> None:
        """Building of table rate limiter."""
        pass

    def on_building_tables_make_tables(self, builder: SystemBuilder) -> None:
        """Building data server table."""
        pass

    def on_building_tables_end(self, builder: SystemBuilder) -> None:
        """End of data server table building."""
        pass

    # PARAMETER SERVER
    def on_building_parameter_server_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_parameter_server_make_parameter_server(
        self, builder: SystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_parameter_server_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # EXECUTOR
    def on_building_adder_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_adder_set_priority(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_adder_make_adder(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_adder_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_parameter_client(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_make_executor(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # TRAINER
    def on_building_dataset_start(self, builder: SystemBuilder) -> None:
        """Start of trainer dataset building."""
        pass

    def on_building_dataset_make_dataset(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_dataset_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_dataset(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_parameter_client(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_make_trainer(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # BUILD
    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    # LAUNCH
    def on_building_launch(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass
