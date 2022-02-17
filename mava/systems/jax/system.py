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

# TODO(Arnu): remove later
# type: ignore

"""Jax-based Mava system implementation."""

from types import SimpleNamespace
from typing import Any, List

from mava.core_jax import BaseSystem
from mava.systems.jax import Builder, Config, Design


# TODO(Arnu): replace component types with Callback when class is ready.
class System(BaseSystem):
    def __init__(self, design: Design) -> None:
        """_summary_

        Args:
            design : _description_
        """
        self._design = design
        self.config = Config()  # Mava config
        self.components: List = []

        # make config from build
        self._make_config()

    def _make_config(self) -> None:
        """_summary_"""
        for component in self._design.components:
            comp = component()
            # note this requires default dataclasses for components
            # note this requires "name" properties for components
            input = {comp.name: comp.config}
            self.config.add(**input)
        self.config.build()

    def update(self, component: Any, name: str) -> None:
        """Update a component that has already been added to the system.

        Args:
            component : system callback component
            name : component name
        """

        if name in list(self._design.components.__dict__.keys()):
            self._design.components.__dict__[name] = component
            self.config.update(name=component().config)
        else:
            raise Exception(
                "The given component is not part of the current system.\
                Perhaps try adding it instead using .add()."
            )

    def add(self, component: Any, name: str) -> None:
        """Add a new component to the system.

        Args:
            component : system callback component
            name : component name
        """

        if name in list(self._design.components.__dict__.keys()):
            raise Exception(
                "The given component is already part of the current system.\
                Perhaps try updating it instead using .update()."
            )
        else:
            self._design.components.__dict__[name] = component
            self.config.add(name=component().config)

    def launch(
        self,
        config: SimpleNamespace,
        num_executors: int,
        nodes_on_gpu: List[str],
        multi_process: bool = True,
        name: str = "system",
    ) -> None:
        """Run the system.

        Args:
            config : system configuration including
            num_executors : number of executor processes to run in parallel
            nodes_on_gpu : which processes to run on gpu
            multi_process : whether to run locally or distributed, local runs are
                for debugging
            name : name of the system
        """
        # update distributor config
        config.set(
            num_executors=num_executors,
            nodes_on_gpu=nodes_on_gpu,
            multi_process=multi_process,
            name=name,
        )
        system_config = config.get()

        # update default system component configs
        for component in self._design.components:
            self.components.append(component(system_config))

        # Build system
        self._builder = Builder(components=self.components)
        self._builder.build()

        # Launch system
        self._builder.launch()
