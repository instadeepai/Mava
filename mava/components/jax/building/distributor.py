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
from typing import Callable, List, Optional, Union

import jax
from chex import PRNGKey

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
    rng_seed: int = 0


class Distributor(Component):
    def __init__(self, config: DistributorConfig = DistributorConfig()):
        """_summary_

        Args:
            config : _description_.
        """
        if isinstance(config.nodes_on_gpu, str):
            config.nodes_on_gpu = [config.nodes_on_gpu]
        self.config = config

    def on_building_program_nodes(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        builder.store.program = Launcher(
            multi_process=self.config.multi_process,
            nodes_on_gpu=self.config.nodes_on_gpu,
            name=self.config.distributor_name,
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
        distributor_key = jax.random.PRNGKey(self.config.rng_seed)
        distributor_key, executor_seed_key, evaluator_seed_key = jax.random.split(
            distributor_key, 3
        )
        executor_seeds = jax.random.randint(
            executor_seed_key,
            (self.config.num_executors,),
            minval=-jax.numpy.inf,
            maxval=jax.numpy.inf,
        )

        # executor nodes
        for executor_id in range(self.config.num_executors):
            builder.store.program.add(
                builder.executor,
                [
                    f"executor_{executor_id}",
                    data_server,
                    parameter_server,
                    executor_seeds[executor_id],
                ],
                node_type=NodeType.corrier,
                name="executor",
            )

        if self.config.run_evaluator:
            evaluator_seed = jax.random.randint(
                evaluator_seed_key, (), minval=-jax.numpy.inf, maxval=jax.numpy.inf
            )
            # evaluator node
            builder.store.program.add(
                builder.executor,
                ["evaluator", data_server, parameter_server, evaluator_seed],
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

        if not self.config.multi_process:
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
