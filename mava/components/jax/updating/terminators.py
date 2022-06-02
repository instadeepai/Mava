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
from typing import Any

from mava.components.jax.component import Component


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
    def on_termination_update(
        self,
    ) -> None:
        pass

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "temination_condition"
