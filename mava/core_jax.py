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
    def design(self) -> SimpleNamespace:
        """System design specifying the list of components to use.

        Returns:
            system callback components
        """

    @abc.abstractmethod
    def update(self, component: Any) -> None:
        """Update a component that has already been added to the system.

        Args:
            component : system callback component
        """

    @abc.abstractmethod
    def add(self, component: Any) -> None:
        """Add a new component to the system.

        Args:
            component : system callback component
        """

    @abc.abstractmethod
    def configure(self, **kwargs: Any) -> None:
        """Configure system hyperparameters."""

    @abc.abstractmethod
    def launch(
        self,
        num_executors: int,
        nodes_on_gpu: List[str],
        multi_process: bool = True,
        name: str = "system",
    ) -> None:
        """Run the system.

        Args:
            num_executors : number of executor processes to run in parallel
            nodes_on_gpu : which processes to run on gpu
            multi_process : whether to run single or multi process, single process runs
                are primarily for debugging
            name : name of the system
        """


class SystemBuilder(abc.ABC):
    """Abstract system builder."""

    @abc.abstractmethod
    def data_server(self) -> List[Any]:
        """Data server to store and serve transition data from and to system.

        Returns:
            System data server
        """

    @abc.abstractmethod
    def parameter_server(self) -> Any:
        """Parameter server to store and serve system network parameters.

        Returns:
            System parameter server
        """

    @abc.abstractmethod
    def executor(
        self, executor_id: str, data_server_client: Any, parameter_server_client: Any
    ) -> Any:
        """Executor, a collection of agents in an environment to gather experience.

        Args:
            executor_id : id to identify the executor process for logging purposes
            data_server_client : data server client for pushing transition data
            parameter_server_client : parameter server client for pulling parameters
        Returns:
            System executor
        """

    @abc.abstractmethod
    def trainer(
        self, trainer_id: str, data_server_client: Any, parameter_server_client: Any
    ) -> Any:
        """Trainer, a system process for updating agent specific network parameters.

        Args:
            trainer_id : id to identify the trainer process for logging purposes
            data_server_client : data server client for pulling transition data
            parameter_server_client : parameter server client for pushing parameters
        Returns:
            System trainer
        """

    @abc.abstractmethod
    def build(self) -> None:
        """Construct program nodes."""

    @abc.abstractmethod
    def launch(self) -> None:
        """Run the graph program."""
