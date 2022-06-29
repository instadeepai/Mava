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
import time
from typing import Any, Callable, Dict, List, Optional, Type

import launchpad as lp
from chex import dataclass

from mava.callbacks import Callback
from mava.components.jax.component import Component
from mava.components.jax.updating.parameter_server import ParameterServer
from mava.core_jax import SystemParameterServer
from mava.utils.training_utils import check_count_condition


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
        return "termination_condition"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        None required.

        Returns:
            List of required component classes.
        """
        return []


@dataclass
class CountConditionTerminatorConfig:
    termination_condition: Optional[Dict[str, Any]] = None
    termination_function: Callable = lp.stop


class CountConditionTerminator(Terminator):
    def __init__(
        self,
        config: CountConditionTerminatorConfig = CountConditionTerminatorConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

        if self.config.termination_condition is not None:
            self.termination_key, self.termination_value = check_count_condition(
                self.config.termination_condition
            )

    def on_parameter_server_run_loop_termination(
        self,
        parameter_sever: SystemParameterServer,
    ) -> None:
        """_summary_"""
        if (
            self.config.termination_condition is not None
            and parameter_sever.store.parameters[self.termination_key]
            > self.termination_value
        ):
            print(
                f"Max {self.termination_key} of {self.termination_value}"
                " reached, terminating."
            )
            self.config.termination_function()

    @staticmethod
    def config_class() -> Type[CountConditionTerminatorConfig]:
        """_summary_"""
        return CountConditionTerminatorConfig

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        ParameterServer required to set parameter_sever.store.parameters.

        Returns:
            List of required component classes.
        """
        return Terminator.required_components() + [ParameterServer]


@dataclass
class TimeTerminatorConfig:
    run_seconds: float = 60.0
    termination_function: Callable = lp.stop


class TimeTerminator(Terminator):
    def __init__(
        self,
        config: TimeTerminatorConfig = TimeTerminatorConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config
        self._start_time = 0.0

    def on_parameter_server_init(self, parameter_sever: SystemParameterServer) -> None:
        """_summary_"""
        self._start_time = time.time()

    def on_parameter_server_run_loop_termination(
        self, parameter_sever: SystemParameterServer
    ) -> None:
        """_summary_"""
        if time.time() - self._start_time > self.config.run_seconds:
            print(
                f"Run time of {self.config.run_seconds} seconds reached, terminating."
            )
            self.config.termination_function()

    @staticmethod
    def config_class() -> Type[TimeTerminatorConfig]:
        """_summary_"""
        return TimeTerminatorConfig
