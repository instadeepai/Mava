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

from mava.systems.building import BaseSystemBuilder


class Callback(abc.ABC):
    """
    Abstract base class used to build new callbacks.
    Subclass this class and override any of the relevant hooks
    """

    ######################
    # system builder hooks
    ######################

    def on_building_init_start(self, builder: BaseSystemBuilder) -> None:
        """Called when the builder initialisation begins."""
        pass

    def on_building_init_end(self, builder: BaseSystemBuilder) -> None:
        """Called when the builder initialisation ends."""
        pass

    def on_building_make_replay_table_start(self, builder: BaseSystemBuilder) -> None:
        """[description]"""
        pass

    def on_building_adder_signature(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_rate_limiter(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_tables(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_replay_table_end(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_dataset_iterator_start(
        self, builder: BaseSystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_dataset(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_dataset_iterator_end(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_adder_start(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_adder_priority(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_adder(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_tables(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_replay_table_start(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_variable_server_start(
        self, builder: BaseSystemBuilder
    ) -> None:
        """[summary]"""
        pass

    def on_building_variable_server(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_variable_server_end(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_executor_start(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_variable_client(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_executor(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_executor_end(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_trainer_start(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_variable_client(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_trainer_statistics(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_make_trainer_end(self, builder: BaseSystemBuilder) -> None:
        """[summary]"""
        pass
