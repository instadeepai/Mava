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


"""Core Mava interfaces for Jax systems."""

import abc
from types import SimpleNamespace
from typing import Any, List


class BaseSystem(abc.ABC):
    """Abstract system object."""

    @abc.abstractmethod
    def configure(self, config: Any) -> SimpleNamespace:
        """[summary]

        Args:
            config ([type]): [description]

        Returns:
            [type]: [description]
        """

    @abc.abstractmethod
    def update(self, component: Any, name: str) -> None:
        """[summary]

        Args:
            component (Any): [description]
            name (str): [description]
        """

    @abc.abstractmethod
    def add(self, component: Any, name: str) -> None:
        """[summary]

        Args:
            component (Any): [description]
            name (str): [description]
        """

    @abc.abstractmethod
    def launch(
        self,
        num_executors: int,
        multi_process: str,
        nodes_on_gpu: List[str],
        name: str,
    ):
        """[summary]

        Args:
            num_executors (int): [description]
            multi_process (str): [description]
            nodes_on_gpu (List[str]): [description]
            name (str): [description]
        """