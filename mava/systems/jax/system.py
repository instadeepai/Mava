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

"""Jax-based Mava system implementation."""
import abc
import copy
from types import SimpleNamespace
from typing import Any, Dict, List, Tuple, Type

from mava.components.jax import Component
from mava.core_jax import BaseSystem
from mava.specs import DesignSpec
from mava.systems.jax import Builder, Config


# TODO(Arnu): replace component types with Callback when class is ready.
class System(BaseSystem):
    """General Mava system class for Jax-based systems."""

    def __init__(self) -> None:
        """System Initialisation.

        Creates config from the build and checks that component keys
        match their assigned names.
        """
        self._design, self._default_params = self.design()

        self.config = Config()  # Mava config
        self.components: List = []
        self._built = False

        # make config from build
        self._make_config()

        # Enforce that design keys match component names
        for key, value in self._design.get().items():
            if key != value.name():
                raise Exception(
                    f"Component '{key}' has mismatching name '{value.name()}'"
                )

    def _make_config(self) -> None:
        """Private method to construct system config upon initialisation.

        Returns:
            None.
        """
        for component in self._design.get().values():
            config_class = component.__init__.__annotations__["config"]

            if config_class is not SimpleNamespace:
                input = {component.name(): config_class()}
                self.config.add(**input)

    @abc.abstractmethod
    def design(self) -> Tuple[DesignSpec, Dict]:
        """System design specifying the list of components to use.

        Returns:
            Tuple[system callback components, system config].
        """

    def update(self, component: Any) -> None:
        """Update a component that has already been added to the system.

        Args:
            component: system callback component.

        Returns:
            None.
        """
        if self._built:
            raise Exception(
                "System already built. Must call .update() on components before the \
                    system has been built."
            )
        name = component.name()

        if name in list(self._design.get().keys()):
            self._design.get()[name] = component
            config_class = component.__init__.__annotations__["config"]
            if config_class:
                config_feed = {name: config_class()}
                self.config.update(**config_feed)
        else:
            raise Exception(
                f"The given component ({name}) is not part of the current system.\
                Perhaps try adding it instead using .add()."
            )

    def add(self, component: Any) -> None:
        """Add a new component to the system.

        Args:
            component: system callback component.

        Returns:
            None.
        """
        if self._built:
            raise Exception(
                "System already built. Must call .add() on components before the \
                    system has been built."
            )
        name = component.name()
        if name in list(self._design.get().keys()):
            raise Exception(
                "The given component is already part of the current system.\
                Perhaps try updating it instead using .update()."
            )
        else:
            self._design.get()[name] = component
            config_class = component.__init__.__annotations__["config"]
            if config_class:
                config_feed = {name: config_class()}
                self.config.add(**config_feed)

    def build(self, **kwargs: Any) -> None:
        """Build the system with all components and hyperparameters.

        Args:
            **kwargs: dictionary {config name: value} to replace system config values.

        Returns:
            None.
        """

        if self._built:
            raise Exception("System already built.")

        # Add the system defaults, but allow the kwargs to overwrite them.
        if self._default_params:
            parameter = copy.copy(self._default_params.__dict__)
        else:
            parameter = {}
        parameter.update(kwargs)

        self.config.build()

        self.config.set_parameters(**parameter)

        # get system config to feed to component list to update hyperparameter settings
        system_config = self.config.get()

        # update default system component configs
        assert len(self.components) == 0
        component: Type[Component]  # provide type
        for component in self._design.get().values():
            self.components.append(
                component(config=self.config.get_local_config(component))
            )

        # Build system
        self._builder = Builder(components=self.components, global_config=system_config)
        self._builder.build()
        self._built = True

    def launch(self) -> None:
        """Run the system by launching the builder.

        Raises:
            Exception: if system has not already been built.

        Returns:
            None.
        """
        if not self._built:
            raise Exception(
                "System not built. First call .build() before calling .launch()."
            )

        # Launch system
        self._builder.launch()

        if self._builder.store.worker_manager:
            return self._builder.store.worker_manager
