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

from typing import Any
from types import SimpleNamespace

from mava.callbacks.base import Callback
from mava.core import BaseSystem
from mava.systems.building import Builder
from mava.components import building


class System(BaseSystem):
    def __init__(self, config):

        self._config = config
        self._distribute = False
        self.system_components = self.configure(self._config)
        self.component_names = list(self.system_components.__dict__.keys())

    @abc.abstractmethod
    def configure(self, config: Any) -> SimpleNamespace:
        """[summary]"""

    def update(self, component: Callback, name: str):
        if name in self.component_names:
            self.system_components.__dict__[name] = component
        else:
            raise Exception(
                "The given component is not part of the current system. Perhaps try adding it instead using .add()."
            )

    def add(self, component: Callback, name: str):
        if name in self.component_names:
            raise Exception(
                "The given component is already part of the current system. Perhaps try updating it instead using .update()."
            )
        else:
            self.system_components.__dict__[name] = component

    def build(self, name="system"):
        self._name = name
        self._component_feed = list(self.system_components)

        # Builder
        self._builder = Builder(components=self.system_components)
        self._builder.build()

    def distribute(self, num_executors=1, nodes_on_gpu=["trainer"]):
        self._distribute = True

        # Distributor
        distributor = building.Distributor(
            num_executors=num_executors,
            multi_process=True,
            nodes_on_gpu=nodes_on_gpu,
            name=self._name,
        )
        self._system_components.append(distributor)

    def launch(self):
        if not self._distribute:
            distributor = building.Distributor(multi_process=False)
            self._system_components.append(distributor)

        self._builder.launch()