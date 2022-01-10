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

from typing import Dict, Any, List

from mava.core import SystemBuilder
from mava.systems.executing import Executor as Exec
from mava.callbacks import Callback
from mava.wrappers import DetailedPerAgentStatistics


class Executor(Callback):
    def __init__(
        self, config: Dict[str, Any], components: List[Callback], evaluator=False
    ):
        """[summary]

        Args:
            config (Dict[str, Any]): [description]
            components (List[Callback]): [description]
            evaluator (bool, optional): [description]. Defaults to False.
        """
        self.config = config
        self.components = components

    def on_building_executor_make_executor(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # create networks
        networks = builder.system()

        # create adder
        adder = builder.adder(builder._replay_client)

        # update config
        self.config.update(
            {
                "networks": networks,
                "adder": adder,
                "variable_client": builder.executor_variable_client,
                "counts": builder.executor_counts,
                "logger": builder.executor_logger,
            }
        )

        builder.executor = Exec(self.config, self.components)

    def on_building_executor_train_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # Create the loop to connect environment and executor.
        builder.train_loop = builder._train_loop_fn(
            builder.executor_environment,
            builder.executor,
            logger=builder.executor_logger,
            **builder._train_loop_fn_kwargs,
        )

    def on_building_executor_end(self, builder: SystemBuilder) -> None:
        """[summary]"""

        builder.train_loop = DetailedPerAgentStatistics(builder.train_loop)

    def on_building_evaluator_make_evaluator(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # create networks
        networks = builder.system()

        # TODO(Arnu): create variable client for evaluator
        # update config
        self.config.update(
            {
                "networks": networks,
                "adder": adder,
                "variable_client": builder.executor_variable_client,
                "counts": builder.executor_counts,
                "logger": builder.executor_logger,
            }
        )

        builder.evaluator = Exec(self.config, self.components)

    def on_building_evaluator_eval_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # Create the loop to connect environment and executor.
        builder.evaluator_loop = builder._evaluator_loop_fn(
            builder.evaluator_environment,
            builder.evaluator,
            logger=builder.evaluator_logger,
            **builder._eval_loop_fn_kwargs,
        )

    def on_building_evaluator_end(self, builder: SystemBuilder) -> None:
        """[summary]"""

        builder.evaluator_loop = DetailedPerAgentStatistics(builder.evaluator_loop)
