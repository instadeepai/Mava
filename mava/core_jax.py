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
from typing import Any, Dict, List, Sequence, Union


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

    def __init__(
        self,
    ) -> None:
        """System building init"""

        # Simple namespace for assigning system builder attributes dynamically
        self.attr = SimpleNamespace()

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


class SystemParameterServer(abc.ABC):
    def __init__(
        self,
    ) -> None:
        """System parameter server init"""

        # Simple namespace for assigning system parameter server attributes dynamically
        self.attr = SimpleNamespace()

    @abc.abstractmethod
    def get_parameters(
        self, names: Union[str, Sequence[str]]
    ) -> Dict[str, Dict[str, Any]]:
        """Get parameters from the parameter server.

        Args:
            names : Names of the parameters to get
        Returns:
            The parameters that were requested
        """

    @abc.abstractmethod
    def set_parameters(self, names: Sequence[str], vars: Dict[str, Any]) -> None:
        """Set parameters in the parameter server.

        Args:
            names : Names of the parameters to set
            vars : The values to set the parameters to
        """

    @abc.abstractmethod
    def add_to_parameters(self, names: Sequence[str], vars: Dict[str, Any]) -> None:
        """Add to the parameters in the parameter server.

        Args:
            names : Names of the parameters to add to
            vars : The values to add to the parameters to
        """

    def run(self) -> None:
        """Run the parameter server. This function allows for checkpointing and other \
        centralised computations to be performed by the parameter server."""
