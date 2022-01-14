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
from mava.utils.decorators import execution
from mava.utils.loggers import MavaLogger


class Logger(Callback):
    def __init__(
        self,
        logger_factory: Callable[[str], MavaLogger] = None,
        logger_config: Dict = {},
    ):
        """[summary]"""
        self.logger_factory = logger_factory
        self.logger_config = logger_config
        self.evaluation = True

    # TODO(Arnu): not sure if this is the cleanest way.
    def make_logger(self, name: str, builder: SystemBuilder) -> MavaLogger:
        logger_config = {}
        if self.logger_config and name in self.logger_config:
            logger_config = self.logger_config[name]
        if name == "executor":
            logger = self.logger_factory(  # type: ignore
                f"{name}_{builder._executor_id}", **logger_config
            )
        elif name == "evaluator":
            logger = self.logger_factory(name, **logger_config)  # type: ignore
        elif name == "trainer":
            logger = self.logger_factory(  # type: ignore
                f"{name}_{builder._trainer_id}", **logger_config
            )
        else:
            raise NotImplementedError

        return logger

    def on_building_executor_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # Create executor logger
        builder.executor_logger = self.make_logger("executor")

    def on_building_evaluator_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # Create evaluator logger
        builder.evaluator_logger = self.make_logger("evaluator")

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # Create trainer logger
        builder.trainer_logger = self.make_logger("trainer")