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

from mava.callbacks import Callback
from mava.wrappers import DetailedPerAgentStatistics


class Executor(Callback):
    def __init__(self, config: Dict[str, Any], components: List[Callback]):
        """[summary]

        Args:
            executor_fn (Type[core.Executor]): [description]
        """
        self.config = config
        self.components = components

    def on_building_init(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # TODO(Arnu): need to setup executor samples/ network sampling setup

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # Create executor logger
        executor_logger_config = {}
        if builder._logger_config and "executor" in builder._logger_config:
            executor_logger_config = builder._logger_config["executor"]
        exec_logger = builder._logger_factory(  # type: ignore
            f"executor_{builder._executor_id}", **executor_logger_config
        )
        builder.executor_logger = exec_logger

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

        builder.executor = Executor(self.config, self.components)

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        builder.executor_environment = builder.environment_factory(evaluation=False)  # type: ignore

    def on_building_executor_train_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # Create the loop to connect environment and executor.
        builder.train_loop = builder._train_loop_fn(
            builder.executor_environment,
            builder.executor,
            logger=builder.executor_logger,
            **builder._train_loop_fn_kwargs,
        )

    def on_building_evaluator_end(self, builder: SystemBuilder) -> None:
        """[summary]"""

        builder.train_loop = DetailedPerAgentStatistics(builder.train_loop)


class Evaluator(Executor):
    def on_building_evaluator_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # Create eval logger.
        evaluator_logger_config = {}
        if builder._logger_config and "evaluator" in builder._logger_config:
            evaluator_logger_config = builder._logger_config["evaluator"]
        eval_logger = builder._logger_factory(  # type: ignore
            "evaluator", **evaluator_logger_config
        )
        builder._eval_logger = eval_logger

    def on_building_evaluator_make_evaluator(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # create networks
        networks = builder.system()

        # create adder
        adder = builder.adder(builder._replay_client)

        # TODO(Arnu): create variable client for evaluator

        builder.evaluator = Executor(self.config, self.components)

    def on_building_evaluator_environment(self, builder: SystemBuilder) -> None:
        builder.evaluator_environment = builder.environment_factory(evaluation=False)  # type: ignore

    def on_building_evaluator_eval_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # Create the loop to connect environment and executor.
        builder.eval_loop = builder._eval_loop_fn(
            builder.evaluator_environment,
            builder.evaluator,
            logger=builder.evaluator_logger,
            **builder._eval_loop_fn_kwargs,
        )

    def on_building_evaluator_end(self, builder: SystemBuilder) -> None:
        """[summary]"""

        builder.eval_loop = DetailedPerAgentStatistics(builder.eval_loop)