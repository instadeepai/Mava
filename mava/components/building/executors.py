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
from mava.systems.executing import Executor as Execute
from mava.callbacks import Callback
from mava.utils.decorators import execution, evaluation


class Executor(Callback):
    def __init__(
        self, config: Dict[str, Any], components: List[Callback], evaluation=False
    ):
        """[summary]

        Args:
            config (Dict[str, Any]): [description]
            components (List[Callback]): [description]
            evaluator (bool, optional): [description]. Defaults to False.
        """
        self.config = config
        self.components = components
        self.evaluation = evaluation

    @execution
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

        builder.executor = Execute(self.config, self.components, self.evaluation)

    @evaluation
    def on_building_evaluator_make_evaluator(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # create networks
        networks = builder.system()

        # update config
        self.config.update(
            {
                "networks": networks,
                "variable_client": builder.evaluator_variable_client,
                "counts": builder.evaluator_counts,
                "logger": builder.evaluator_logger,
            }
        )

        builder.evaluator = Execute(self.config, self.components, self.evaluation)
