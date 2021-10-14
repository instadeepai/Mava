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

"""Commonly used dataset components for system builders"""
import launchpad as lp

from mava.callbacks import Callback
from mava.systems.building import SystemBuilder


class Distributor(Callback):
    def on_building_distributor_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_distributor_tables(self, builder: SystemBuilder) -> None:
        """[summary]"""
        with builder.program.group("tables"):
            builder.program_tables = builder.program.add_node(
                lp.ReverbNode(builder.tables)
            )

    def on_building_distributor_variable_server(self, builder: SystemBuilder) -> None:
        """[summary]"""
        with builder.program.group("variable_server"):
            builder.program_variable_server = builder.program.add_node(
                lp.CourierNode(builder.variable_server)
            )

    def on_building_distributor_trainer(self, builder: SystemBuilder) -> None:
        """[summary]"""
        with builder.program.group("trainer"):
            # Add executors which pull round-robin from our variable sources.
            for trainer_id in range(len(builder.config.trainer_networks.keys())):
                builder.program.add_node(
                    lp.CourierNode(
                        builder.trainer,
                        trainer_id,
                        builder.program_tables,
                        builder.program_variable_server,
                    )
                )

    def on_building_distributor_evaluator(self, builder: SystemBuilder) -> None:
        """[summary]"""
        with builder.program.group("evaluator"):
            builder.program.add_node(
                lp.CourierNode(builder.evaluator, builder.program_variable_server)
            )

    def on_building_distributor_executor(self, builder: SystemBuilder) -> None:
        """[summary]"""
        with builder.program.group("executor"):
            # Add executors which pull round-robin from our variable sources.
            for executor_id in range(builder._num_exectors):
                builder.program.add_node(
                    lp.CourierNode(
                        builder.executor,
                        executor_id,
                        builder.program_tables,
                        builder.program_variable_server,
                    )
                )

    def on_building_distributor_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass
