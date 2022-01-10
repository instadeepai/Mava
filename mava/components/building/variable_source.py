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

"""Commonly used adder signature components for system builders"""
import abc

from mava.callbacks import Callback
from mava.core import SystemBuilder


class VariableSource(Callback):
    @abc.abstractmethod
    def on_building_variable_server(self, builder: SystemBuilder) -> None:
        """[summary]"""

    @abc.abstractmethod
    def on_building_executor_variable_client(self, builder: SystemBuilder) -> None:
        """[summary]"""

    @abc.abstractmethod
    def on_building_trainer_variable_client(self, builder: SystemBuilder) -> None:
        """[summary]"""