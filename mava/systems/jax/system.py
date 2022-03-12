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
from typing import Any, Callable, List

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
        comp = component()
        name = comp.name
        if name in list(self._design.__dict__.keys()):
            self._design.__dict__[name] = component
            config_feed = {name: comp.config}
            self.config.update(**config_feed)
        else:
            raise Exception(
                "The given component is not part of the current system.\
                Perhaps try adding it instead using .add()."
            )

    def add(self, component: Any) -> None:
        """Add a new component to the system.

        Args:
            component : system callback component
            name : component name
        """
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

    def configure(self, **kwargs: Any) -> None:
        """Configure system hyperparameters."""
        self.config.build()
        self.config.set_parameters(**kwargs)

    def launch(
        self,
        num_executors: int,
        nodes_on_gpu: List[str],
        multi_process: bool = True,
        name: str = "system",
        builder_class: Callable = Builder,
    ) -> None:
        """Run the system.

        Args
            num_executors : number of executor processes to run in parallel
            nodes_on_gpu : which processes to run on gpu
            multi_process : whether to run locally or distributed, local runs are
                for debugging
            name : name of the system
            builder_class: callable builder class.
        """
        # build config is not already built
        if not self.config._built:
            self.config.build()

        # update distributor config
        self.config.set_parameters(
            num_executors=num_executors,
            nodes_on_gpu=nodes_on_gpu,
            multi_process=multi_process,
            name=name,
        )

        # get system config to feed to component list to update hyperparamete settings
        system_config = self.config.get()

        # update default system component configs
        for component in self._design.__dict__.values():
            self.components.append(component(system_config))

        # Build system
        self._builder = builder_class(components=self.components)
        self._builder.build()

        # Launch system
        self._builder.launch()
