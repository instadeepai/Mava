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

"""Mava variable server implementation."""


import abc
from typing import Dict, Any

from mava.callbacks import Callback


class VariableServer(Callback):
    def __init__(
        self,
        variables: Dict[str, Any],
    ) -> None:
        """Initialise the variable server"""

        self.variables = variables

    @abc.abstractmethod
    def on_variables_get_server_variables(self) -> None:
        """[summary]"""

    @abc.abstractmethod
    def on_variables_set_server_variables_if_tuple(self) -> None:
        """[summary]"""

    @abc.abstractmethod
    def on_variables_set_server_variables_if_dict(self) -> None:
        """[summary]"""

    @abc.abstractmethod
    def on_variables_add_to_server_variables(self) -> None:
        """[summary]"""

    @abc.abstractmethod
    def on_variables_run_server_loop(self) -> None:
        """[summary]"""
