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
from typing import Any, Callable, Dict, Optional, Type

import launchpad as lp
from chex import dataclass

from mava.components.jax.component import Component
from mava.core_jax import SystemParameterServer
from mava.utils.training_utils import check_count_condition


class Terminator(Component):
    @abc.abstractmethod
    def __init__(
        self,
        config: Any,
    ):
        """Component handles when a system / run should terminate."""
        self.config = config

    @abc.abstractmethod
    def on_parameter_server_run_loop_termination(
        self, parameter_sever: SystemParameterServer
    ) -> None:
        """Terminate system if some condition is met."""
        pass

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "termination_condition"


@dataclass
class CountConditionTerminatorConfig:
    termination_condition: Optional[Dict[str, Any]] = None
    termination_function: Callable = lp.stop


class CountConditionTerminator(Terminator):
    def __init__(
        self,
        config: CountConditionTerminatorConfig = CountConditionTerminatorConfig(),
    ):
        """Component terminates a run when a count parameter reaches a threshold.

        Args:
            config: CountConditionTerminatorConfig.
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
        """Terminate a run when a parameter exceeds the given threshold.

        Args:
            parameter_sever: SystemParameterServer.

        Returns:
            None.
        """
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
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return CountConditionTerminatorConfig


@dataclass
class TimeTerminatorConfig:
    run_seconds: float = 60.0
    termination_function: Callable = lp.stop


class TimeTerminator(Terminator):
    def __init__(
        self,
        config: TimeTerminatorConfig = TimeTerminatorConfig(),
    ):
        """Component terminates a run when it reaches a time limit.

        Args:
            config: TimeTerminatorConfig.
        """
        self.config = config
        self._start_time = 0.0

    def on_parameter_server_init(self, parameter_sever: SystemParameterServer) -> None:
        """Store the time at which the system was initialised.

        Args:
            parameter_sever: SystemParameterServer.

        Returns:
            None.
        """
        self._start_time = time.time()

    def on_parameter_server_run_loop_termination(
        self, parameter_sever: SystemParameterServer
    ) -> None:
        """Terminate the system if the time elapsed has exceeded the limit.

        Args:
            parameter_sever: SystemParameterServer.

        Returns:
            None.
        """
        if time.time() - self._start_time > self.config.run_seconds:
            print(
                f"Run time of {self.config.run_seconds} seconds reached, terminating."
            )
            self.config.termination_function()

    @staticmethod
    def config_class() -> Type[TimeTerminatorConfig]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return TimeTerminatorConfig
