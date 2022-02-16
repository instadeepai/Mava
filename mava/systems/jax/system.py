# # python3
# # Copyright 2021 InstaDeep Ltd. All rights reserved.
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #     http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.

# """Mava system implementation."""

# import abc
# from types import SimpleNamespace
# from typing import Any, List

# from mava.callbacks import Callback
# from mava.components import building
# from mava.core import BaseSystem
# from mava.systems import Builder


# class System(BaseSystem):
#     def update(self, component: Callback, name: str) -> None:
#         """Update a component that has already been added to the system.
#         Args:
#             component : system callback component
#             name : component name
#         """

#         if name in list(self.system_components.__dict__.keys()):
#             self.system_components.__dict__[name] = component
#         else:
#             raise Exception(
#                 "The given component is not part of the current system.
# Perhaps try adding it instead using .add()."
#             )

#     def add(self, component: Callback, name: str) -> None:
#         """Add a new component to the system.
#         Args:
#             component : system callback component
#             name : component name
#         """

#         if name in list(self.system_components.__dict__.keys()):
#             raise Exception(
#                 "The given component is already part of the current system.
# Perhaps try updating it instead using .update()."
#             )
#         else:
#             self.system_components.__dict__[name] = component

#     @abc.abstractmethod
#     def build(self, config: SimpleNamespace) -> SimpleNamespace:
#         """Build system by constructing system components.
#         Args:
#             config : system configuration including
#         Returns:
#             System components
#         """

#     def distribute(
#         self,
#         num_executors: int = 1,
#         nodes_on_gpu: List[str] = ["trainer"],
#         distributor: Callback = None,
#     ):
#         """Distribute system across multiple processes.

#         Args:
#             num_executors : number of executor processes to run in parallel
#             multi_process : whether to run locally or distributed
#             nodes_on_gpu : which processes to run on gpu
#             distributor : custom distributor component
#         """

#         # Distributor
#         distributor_fn = distributor if distributor else building.Distributor
#         distribute = distributor_fn(
#             num_executors=num_executors,
#             nodes_on_gpu=nodes_on_gpu,
#             run_evaluator="evaluator" in list(self.system_components.__dict__.keys()),
#         )
#         self.add(component=distribute, name="distributor")

#     def launch(
#         self,
#         name: str = "system",
#     ):
#         """Run the system.

#         Args:
#             name : name of the system
#         """

#         component_feed = list(self.system_components.__dict__.values())

#         # Builder
#         self._builder = Builder(components=component_feed)
#         self._builder.build()
