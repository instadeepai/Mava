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
from mava.callbacks import Callback
from mava.core import SystemBuilder
from mava.systems.launcher import Launcher, NodeType


class Distributor(Callback):
    def __init__(
        self, num_executors, multi_process=True, nodes_on_gpu=["trainer"], name="System"
    ):
        self._num_executors = num_executors
        # Create the launcher program
        self._program = Launcher(
            multi_process=multi_process, nodes_on_gpu=nodes_on_gpu, name=name
        )

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        builder._program = self._program

        # tables node
        tables = builder._program.add(
            builder.tables, node_type=NodeType.reverb, name="tables"
        )

        # trainer node
        trainer = builder._program.add(
            builder.trainer, tables, node_type=NodeType.corrier, name="trainer"
        )

        # evaluator node
        evaluator = builder._program.add(
            builder.evaluator, trainer, node_type=NodeType.corrier, name="evaluator"
        )

        # executor nodes
        executors = [
            builder._program.add(
                builder.executor,
                [tables, trainer],
                node_type=NodeType.corrier,
                name="executor",
            )
            for _ in range(self._num_executors)
        ]

        # variable server node
        _ = builder._program.add(
            builder.variable_server,
            [trainer, executors, evaluator],
            node_type=NodeType.corrier,
            name="variable_server",
        )

    def on_building_launch_distributor(self, builder: SystemBuilder):
        builder._program.launch()
