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
from typing import Callable, List, Optional, Type, Union

from mava.callbacks import Callback
from mava.components.jax import Component
from mava.components.jax.training.trainer import BaseTrainerInit
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
    single_process_max_episodes: Optional[int] = None
    is_test: Optional[bool] = False


class Distributor(Component):
    def __init__(self, config: DistributorConfig = DistributorConfig()):
        """Component builds launchpad program nodes and launches the program.

        Args:
            config: DistributorConfig.
        """
        if isinstance(config.nodes_on_gpu, str):
            config.nodes_on_gpu = [config.nodes_on_gpu]
        self.config = config

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """Create nodes for the program and save the program in the store.

        Create data server, parameter server, executor, trainer, and evaluator nodes.
        Handles both single-process and multi-process.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        builder.store.program = Launcher(
            multi_process=self.config.multi_process,
            nodes_on_gpu=self.config.nodes_on_gpu,
            name=self.config.distributor_name,
            terminal=self.config.terminal,
            single_process_max_episodes=self.config.single_process_max_episodes,
            is_test=self.config.is_test,
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
            node_type=NodeType.courier,
            name="parameter_server",
        )

        # executor nodes
        for executor_id in range(self.config.num_executors):
            builder.store.program.add(
                builder.executor,
                [f"executor_{executor_id}", data_server, parameter_server],
                node_type=NodeType.courier,
                name="executor",
            )

        if self.config.run_evaluator:
            # evaluator node
            builder.store.program.add(
                builder.executor,
                ["evaluator", data_server, parameter_server],
                node_type=NodeType.courier,
                name="evaluator",
            )

        # trainer nodes
        for trainer_id in builder.store.trainer_networks.keys():
            builder.store.program.add(
                builder.trainer,
                [trainer_id, data_server, parameter_server],
                node_type=NodeType.courier,
                name="trainer",
            )

        if not self.config.multi_process:
            builder.store.system_build = builder.store.program.get_nodes()

    def on_building_launch(self, builder: SystemBuilder) -> None:
        """Start the launchpad program saved in the store.

        Args:
            builder: SystemBuilder.

        Returns:
            None.
        """
        builder.store.program.launch()

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "distributor"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return DistributorConfig

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        BaseTrainerInit required to set up builder.store.trainer_networks.

        Returns:
            List of required component classes.
        """
        return [BaseTrainerInit]
