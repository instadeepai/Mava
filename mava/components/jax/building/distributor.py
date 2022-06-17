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

"""Commonly used distributor components for system builders"""
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Callable, List, Optional, Union

from mava.components.jax import Component
from mava.core_jax import SystemBuilder
from mava.systems.jax.launcher import Launcher, NodeType


@dataclass
class DistributorConfig:
    num_executors: int = 1
    multi_process: bool = True
    nodes_on_gpu: Union[List[str], str] = "trainer"
    run_evaluator: bool = True
    distributor_name: str = "System"
    terminal: str = "current_terminal"


class Distributor(Component):
    def __init__(self,
                 local_config: DistributorConfig = DistributorConfig(),
                 global_config: SimpleNamespace = SimpleNamespace(),
                 ):
        """_summary_

        Args:
            local_config : _description_.
            global_config : _description_.
        """
        if isinstance(local_config.nodes_on_gpu, str):
            local_config.nodes_on_gpu = [local_config.nodes_on_gpu]
        self.local_config = local_config
        self.global_config = global_config

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        builder.store.program = Launcher(
            multi_process=self.local_config.multi_process,
            nodes_on_gpu=self.local_config.nodes_on_gpu,
            name=self.local_config.distributor_name,
            terminal=self.local_config.terminal,
        )

        # tables node
        data_server = builder.store.program.add(
            builder.data_server,
            node_type=NodeType.reverb,
            name="data_server",
        )

        # variable server node
        parameter_server = builder.store.program.add(
            builder.parameter_server,
            node_type=NodeType.corrier,
            name="parameter_server",
        )

        # executor nodes
        for executor_id in range(self.local_config.num_executors):
            builder.store.program.add(
                builder.executor,
                [f"executor_{executor_id}", data_server, parameter_server],
                node_type=NodeType.corrier,
                name="executor",
            )

        if self.local_config.run_evaluator:
            # evaluator node
            builder.store.program.add(
                builder.executor,
                ["evaluator", data_server, parameter_server],
                node_type=NodeType.corrier,
                name="evaluator",
            )

        # trainer nodes
        for trainer_id in builder.store.trainer_networks.keys():
            builder.store.program.add(
                builder.trainer,
                [trainer_id, data_server, parameter_server],
                node_type=NodeType.corrier,
                name="trainer",
            )

        if not self.local_config.multi_process:
            builder.store.system_build = builder.store.program.get_nodes()

    def on_building_launch(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        builder.store.program.launch()

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "distributor"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return DistributorConfig
