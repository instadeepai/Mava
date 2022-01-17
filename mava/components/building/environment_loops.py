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

from typing import Callable, Dict

from mava.core import SystemBuilder

from mava.callbacks import Callback
from mava.environment_loop import ParallelEnvironmentLoop
from mava.wrappers import DetailedPerAgentStatistics
from mava.utils.decorators import execution, evaluation


class EnvironmentLoop(Callback):
    def __init__(
        self,
        environment_loop: Callable = ParallelEnvironmentLoop,
        loop_config: Dict = {},
        evaluation: bool = False,
    ):
        """[summary]"""
        self.environment_loop = environment_loop
        self.loop_config = loop_config
        self.evaluation = evaluation

    @execution
    def on_building_executor_train_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # Create the loop to connect environment and executor.
        builder.train_loop = self._train_loop_fn(
            builder.executor_environment,
            builder.executor,
            logger=builder.executor_logger,
            **builder._train_loop_fn_kwargs,
        )

    @execution
    def on_building_executor_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # TODO (Arnu): find neater and more general way to add trainer loop wrappers
        builder.train_loop = DetailedPerAgentStatistics(builder.train_loop)

    @evaluation
    def on_building_evaluator_eval_loop(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # Create the loop to connect environment and executor.
        builder.evaluator_loop = builder._evaluator_loop_fn(
            builder.evaluator_environment,
            builder.evaluator,
            logger=builder.evaluator_logger,
            **builder._eval_loop_fn_kwargs,
        )

    @evaluation
    def on_building_evaluator_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        builder.evaluator_loop = DetailedPerAgentStatistics(builder.evaluator_loop)