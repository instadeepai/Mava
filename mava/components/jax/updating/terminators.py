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
from typing import Any, Dict, Optional, Type

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


@dataclass
class ParameterServerTerminatorConfig:
    termination_condition: Optional[Dict[str, Any]] = None


class ParameterServerTerminator(Terminator):
    def __init__(
        self,
        config: ParameterServerTerminatorConfig = ParameterServerTerminatorConfig(),
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
        self, parameter_sever: SystemParameterServer
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
            lp.stop()

    @staticmethod
    def config_class() -> Type[ParameterServerTerminatorConfig]:
        """_summary_"""
        return ParameterServerTerminatorConfig
