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

"""Terminator component for Mava systems."""
import abc
from typing import Any, Type

import launchpad as lp
from chex import dataclass

from mava.components.jax.component import Component
from mava.core_jax import SystemParameterServer


class Terminator(Component):
    @abc.abstractmethod
    def __init__(
        self,
        config: Any,
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    @abc.abstractmethod
    def on_parameter_server_run_loop_termination(
        self, parameter_sever: SystemParameterServer
    ) -> None:
        """_summary_"""
        pass

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "temination_condition"


@dataclass
class ParameterTerminatorConfig:
    parameter_termination_key: str = "executor_steps"
    parameter_termination_value: int = 5000


class ParameterTerminator(Terminator):
    def __init__(
        self,
        config: Any = ParameterTerminatorConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

        valid_options = [
            "trainer_steps",
            "trainer_walltime",
            "evaluator_steps",
            "evaluator_episodes",
            "executor_episodes",
            "executor_steps",
        ]
        assert self.config.parameter_termination_key in valid_options, (
            "Please give a valid termination condition. "
            + f"Current valid conditions are {valid_options}"
        )

    def on_parameter_server_run_loop_termination(
        self, parameter_sever: SystemParameterServer
    ) -> None:
        """_summary_"""
        if (
            parameter_sever.store.parameters[self.config.parameter_termination_key]
            > self.config.parameter_termination_value
        ):
            print(
                f"Max {self.config.parameter_termination_key} of "
                + f"{self.config.parameter_termination_value} reached, terminating."
            )
            lp.stop()

    @staticmethod
    def config_class() -> Type[ParameterTerminatorConfig]:
        """_summary_"""
        return ParameterTerminatorConfig
