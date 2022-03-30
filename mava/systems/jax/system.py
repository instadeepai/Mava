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
from types import SimpleNamespace
from typing import Any, List

from mava.core_jax import BaseSystem
from mava.systems.jax import Builder, Config


# TODO(Arnu): replace component types with Callback when class is ready.
class System(BaseSystem):
    """General Mava system class for Jax-based systems."""

    def __init__(self) -> None:
        """System Initialisation"""
        self._design = self.design()
        self.config = Config()  # Mava config
        self.components: List = []
        self._built = False

        # make config from build
        self._make_config()

    def _make_config(self) -> None:
        """Private method to construct system config upon initialisation."""
        for component in self._design.__dict__.values():
            comp = component()
            input = {comp.name: comp.config}
            self.config.add(**input)

    @abc.abstractmethod
    def design(self) -> SimpleNamespace:
        """System design specifying the list of components to use.

        Returns:
            system callback components
        """

    def update(self, component: Any) -> None:
        """Update a component that has already been added to the system.

        Args:
            component : system callback component
            name : component name
        """
        if self._built:
            raise Exception(
                "System already built. Must call .update() on components before the \
                    system has been built."
            )
        comp = component()
        name = comp.name
        if name in list(self._design.__dict__.keys()):
            self._design.__dict__[name] = component
            config_feed = {name: comp.config}
            self.config.update(**config_feed)
        else:
            raise Exception(
                f"The given component ({name}) is not part of the current system.\
                Perhaps try adding it instead using .add()."
            )

    def add(self, component: Any) -> None:
        """Add a new component to the system.

        Args:
            component : system callback component
            name : component name
        """
        if self._built:
            raise Exception(
                "System already built. Must call .add() on components before the \
                    system has been built."
            )
        comp = component()
        name = comp.name
        if name in list(self._design.__dict__.keys()):
            raise Exception(
                "The given component is already part of the current system.\
                Perhaps try updating it instead using .update()."
            )
        else:
            self._design.__dict__[name] = component
            config_feed = {name: comp.config}
            self.config.add(**config_feed)

    def build(self, **kwargs: Any) -> None:
        if self._built:
            raise Exception("System already built.")
        """Configure system hyperparameters."""
        self.config.build()
        self.config.set_parameters(**kwargs)

        # get system config to feed to component list to update hyperparameter settings
        system_config = self.config.get()

        # update default system component configs
        assert len(self.components) == 0
        for component in self._design.__dict__.values():
            self.components.append(component(system_config))

        # Build system
        self._builder = Builder(components=self.components)
        self._builder.build()
        self._built = True

    def launch(self) -> None:
        """Run the system."""
        if not self._built:
            raise Exception(
                "System not built. First call .build() before calling .launch()."
            )

        # Launch system
        self._builder.launch()
