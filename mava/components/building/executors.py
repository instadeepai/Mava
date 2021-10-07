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

"""Execution components for system builders"""

from typing import List, Dict, Type

from acme.specs import EnvironmentSpec

from mava import core
from mava.callbacks import Callback
from mava.systems.building import SystemBuilder

# self.on_building_executor_start(self)

#         self.on_building_executor_logger(self)

#         self.on_building_executor(self)

#         self.on_building_executor_train_loop(self)

#         self.on_building_executor_end(self)

#         # Create the system
#         behaviour_policy_networks, networks = self.create_system()

#         # Create the executor.
#         executor = self._builder.make_executor(
#             networks=networks,
#             policy_networks=behaviour_policy_networks,
#             adder=self._builder.make_adder(replay),
#             variable_source=variable_source,
#         )

#         # TODO (Arnu): figure out why factory function are giving type errors
#         # Create the environment.
#         environment = self._environment_factory(evaluation=False)  # type: ignore

#         # Create executor logger
#         executor_logger_config = {}
#         if self._logger_config and "executor" in self._logger_config:
#             executor_logger_config = self._logger_config["executor"]
#         exec_logger = self._logger_factory(  # type: ignore
#             f"executor_{executor_id}", **executor_logger_config
#         )

#         # Create the loop to connect environment and executor.
#         train_loop = self._train_loop_fn(
#             environment,
#             executor,
#             logger=exec_logger,
#             **self._train_loop_fn_kwargs,
#         )

#         train_loop = DetailedPerAgentStatistics(train_loop)


class Executor(Callback):
    def __init__(
        self,
        executor_fn: Type[core.Executor],
        net_to_ints: Dict[str, int],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        executor_samples: List[str],
    ):
        """[summary]

        Args:
            net_to_ints (Dict[str, int]): [description]
            agent_specs (Dict[str, EnvironmentSpec]): [description]
            agent_net_keys (Dict[str, str]): [description]
            executor_samples (executor_samples): [description]
        """
        self.executor_fn = executor_fn
        self.net_to_ints = net_to_ints
        self.agent_specs = agent_specs
        self.agent_net_keys = agent_net_keys
        self.executor_samples = executor_samples

    def on_building_executor(self, builder: SystemBuilder):
        """[summary]

        Args:
            builder (SystemBuilder): [description]
        """
        builder.executor = self.executor_fn(
            policy_networks=self._policy_networks,
            counts=builder.counts,
            net_to_ints=self.net_to_ints,
            agent_specs=self.agent_specs,
            agent_net_keys=self.agent_net_keys,
            executor_samples=self.executor_samples,
            variable_client=builder.variable_client,
            adder=self._adder,
        )