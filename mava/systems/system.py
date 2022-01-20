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

"""MADDPG system implementation."""
import abc
from types import SimpleNamespace
from typing import Any, List

from mava.callbacks.base import Callback
from mava.components import building
from mava.core import BaseSystem
from mava.systems.building import Builder


class System(BaseSystem):
    def __init__(self, config):

        self._config = config
        self.system_components = self.configure(self._config)

    @abc.abstractmethod
    def configure(self, config: Any) -> SimpleNamespace:
        """[summary]"""

    def update(self, component: Callback, name: str):
        if name in list(self.system_components.__dict__.keys()):
            self.system_components.__dict__[name] = component
        else:
            raise Exception(
                "The given component is not part of the current system. Perhaps try adding it instead using .add()."
            )

    def add(self, component: Callback, name: str):
        if name in list(self.system_components.__dict__.keys()):
            raise Exception(
                "The given component is already part of the current system. Perhaps try updating it instead using .update()."
            )
        else:
            self.system_components.__dict__[name] = component

    def build(
        self,
        num_executors: int = 1,
        multi_process: str = False,
        nodes_on_gpu: List[str] = ["trainer"],
        name: str = "system",
        distributor: Callback = None,
    ):

        # Distributor
        distributor_fn = distributor if distributor else building.Distributor
        distribute = distributor_fn(
            num_executors=num_executors,
            multi_process=multi_process,
            nodes_on_gpu=nodes_on_gpu,
            run_evaluator="evaluator" in list(self.system_components.__dict__.keys()),
            name=name,
        )
        self.add(component=distribute, name="distributor")

        component_feed = list(self.system_components.__dict__.values())

        # Builder
        self._builder = Builder(components=component_feed)
        self._builder.build()

    def launch(self):
        self._builder.launch()
