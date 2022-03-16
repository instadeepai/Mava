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
from typing import List

from mava.callbacks import Callback
from mava.core_jax import SystemBuilder
from mava.systems.jax.launcher import Launcher, NodeType


@dataclass
class DistributorConfig:
    num_executors: int = 1
    multi_process: bool = True
    nodes_on_gpu: List[str] = ["trainer"]
    run_evaluator: bool = True
    distributor_name: str = "System"


class Distributor(Callback):
    def __init__(self, config: DistributorConfig = DistributorConfig()):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        builder.attr.program = Launcher(
            multi_process=self.config.multi_process,
            nodes_on_gpu=self.config.nodes_on_gpu,
            name=self.config.distributor_name,
        )

        # tables node
        data_server = builder.attr.program.add(
            builder.attr.system_data_server,
            node_type=NodeType.reverb,
            name="data_server",
        )

        # variable server node
        parameter_server = builder.attr.program.add(
            builder.attr.system_parameter_server,
            node_type=NodeType.corrier,
            name="parameter_server",
        )

        # trainer nodes
        for trainer_id in builder.attr.trainer_networks.keys():
            builder.attr.program.add(
                builder.attr.system_trainer,
                [trainer_id, data_server, parameter_server],
                node_type=NodeType.corrier,
                name="trainer",
            )

        # executor nodes
        for executor_id in range(self.config.num_executors):
            builder.attr.program.add(
                builder.attr.system_executor,
                [executor_id, data_server, parameter_server],
                node_type=NodeType.corrier,
                name="executor",
            )

        if self.config.run_evaluator:
            # evaluator node
            builder.attr.program.add(
                builder.attr.system_evaluator,
                parameter_server,
                node_type=NodeType.corrier,
                name="evaluator",
            )

    def on_building_launch_distributor(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        builder.attr.program.launch()
