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

from typing import Callable

import dm_env

from mava.core import SystemBuilder

from mava.callbacks import Callback
from mava import specs as mava_specs


class Environment(Callback):
    def __init__(self, environment_factory: Callable[[bool], dm_env.Environment]):
        """[summary]

        Args:
            environment_factory (Callable[[bool], dm_env.Environment]): [description]
        """
        self.environment_factory = environment_factory

    def on_building_init(self, builder: SystemBuilder) -> None:
        """[summary]"""
        builder.environment_factory = self.environment_factory
        builder.environment_spec = mava_specs.MAEnvironmentSpec(
            environment_factory(evaluation=False)  # type: ignore
        )

    def on_building_executor_environment(self, builder: SystemBuilder) -> None:
        builder.executor_environment = self.environment_factory(evaluation=False)  # type: ignore

    def on_building_evaluator_environment(self, builder: SystemBuilder) -> None:
        builder.evaluator_environment = self.environment_factory(evaluation=False)  # type: ignore