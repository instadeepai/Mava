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
    def build(self, config: SimpleNamespace) -> SimpleNamespace:
        """Build system by constructing system components.

        Args:
            config : system configuration including
        Returns:
            System components
        """

    @abc.abstractmethod
    def update(self, component: Any, name: str) -> None:
        """Update a component that has already been added to the system.

        Args:
            component : system callback component
            name : component name
        """

    @abc.abstractmethod
    def add(self, component: Any, name: str) -> None:
        """Add a new component to the system.

        Args:
            component : system callback component
            name : component name
        """

    @abc.abstractmethod
    def launch(
        self,
        num_executors: int,
        multi_process: bool,
        nodes_on_gpu: List[str],
        name: str,
    ) -> None:
        """Run the system, either locally or distributed.

        Args:
            num_executors : number of executor processes to run in parallel.
            multi_process : whether to run locally or distributed
            nodes_on_gpu : which processes to run on gpu
            name : name of the system
        """
