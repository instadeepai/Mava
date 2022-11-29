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
from typing import Any, Dict, List, Sequence, Tuple, Type, Union

# from mava.callbacks.base import Callback
# import mava.callbacks.base as callbacks
from mava.specs import DesignSpec


class BaseSystem(abc.ABC):
    """Abstract system object."""

    @staticmethod
    @abc.abstractmethod
    def design() -> Tuple[DesignSpec, Dict]:
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
    def build(self, **kwargs: Any) -> None:
        """Configure system hyperparameters and build."""

    @abc.abstractmethod
    def launch(self) -> None:
        """Run the system."""


class BaseSystemComponent(abc.ABC):
    """Base class for core components: builder, executor, trainer and dataserver"""

    def __init__(self) -> None:
        """Core component init"""
        self.store: SimpleNamespace
        self.callbacks: List

    def has(self, component: Type, subtypes: bool = False) -> bool:
        """Checks if a component of the given type is in the system.

        Args:
            component: type of component for which to check.
            subtypes: whether or not to check for subtypes of the given component.

        Returns:
            True if the system has a component of this type.
        """
        if subtypes:  # checks for type of component and its subtypes
            return any(
                isinstance(other_component, component)
                for other_component in self.callbacks
            )
        else:  # checks for exactly type of component
            return any(
                type(other_component) == component for other_component in self.callbacks
            )


class SystemBuilder(BaseSystemComponent):
    """Abstract system builder."""

    def __init__(self) -> None:
        """System building init"""
        super().__init__()

        self._executor_id: str
        self._trainer_id: str
        self._data_server_client: Any
        self._parameter_server_client: Any
        self._evaluator: bool

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


class SystemExecutor(BaseSystemComponent):
    """Abstract system executor."""

    def __init__(self) -> None:
        """System executor init"""
        super().__init__()

        self._agent: str
        self._observation: Any
        self._timestep: Any
        self._state: Any
        self._observations: Dict[str, Any]
        self._actions: Dict[str, Any]
        self._extras: Dict[str, Any]

    @abc.abstractmethod
    def select_actions(
        self, observations: Dict[str, Any]
    ) -> Union[Dict[str, Any], Tuple[Dict[str, Any], Dict[str, Any]]]:
        """Select the actions for all agents in the system."""

    @abc.abstractmethod
    def observe(
        self,
        actions: Dict[str, Any],
        timestep: Any,
        extras: Dict[str, Any] = {},
    ) -> None:
        """Record observed timestep from the environment."""

    @abc.abstractmethod
    def update(self, wait: bool = False) -> None:
        """Update executor parameters."""


class SystemTrainer(BaseSystemComponent):
    """Abstract system trainer."""

    def __init__(self) -> None:
        """System trainer init"""
        super().__init__()

        self._inputs: Any

    @abc.abstractmethod
    def step(self) -> None:
        """Trainer forward and backward passes."""


class SystemParameterServer(BaseSystemComponent):
    def __init__(self) -> None:
        """System parameter server init"""
        super().__init__()

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
    def set_parameters(self, set_params: Dict[str, Any]) -> None:
        """Set parameters in the parameter server.

        Args:
            set_params : The values to set the parameters to
        """

    @abc.abstractmethod
    def add_to_parameters(self, add_to_params: Dict[str, Any]) -> None:
        """Add to the parameters in the parameter server.

        Args:
            add_to_params : values to add to the parameters
        """

    def run(self) -> None:
        """Run the parameter server. This function allows for checkpointing and other \
        centralised computations to be performed by the parameter server."""


class SystemParameterClient(BaseSystemComponent):
    """A variable client for updating variables from a remote source."""

    def __init__(self) -> None:
        """System parameter server init"""
        super().__init__()

    @abc.abstractmethod
    def get_async(self) -> None:
        """Asynchronously updates the get variables with the latest copy from source."""

    @abc.abstractmethod
    def set_async(self, params: Dict[str, Any]) -> None:
        """Asynchronously updates source with the set variables."""

    @abc.abstractmethod
    def set_and_get_async(self) -> None:
        """Asynchronously updates source and gets from source."""

    @abc.abstractmethod
    def add_async(self, names: List[str], vars: Dict[str, Any]) -> None:
        """Asynchronously adds to source variables."""

    @abc.abstractmethod
    def add_and_wait(self, names: List[str], vars: Dict[str, Any]) -> None:
        """Adds the specified variables to the corresponding variables in source \
        and waits for the process to complete before continuing."""

    @abc.abstractmethod
    def get_and_wait(self) -> None:
        """Updates the get variables with the latest copy from source \
        and waits for the process to complete before continuing."""

    @abc.abstractmethod
    def get_all_and_wait(self) -> None:
        """Updates all the variables with the latest copy from source \
        and waits for the process to complete before continuing."""

    @abc.abstractmethod
    def set_and_wait(self, params: Dict[str, Any]) -> None:
        """Updates source with the set variables \
        and waits for the process to complete before continuing."""
