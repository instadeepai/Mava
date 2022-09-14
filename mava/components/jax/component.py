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

"""Base components for system builder"""

import abc
from typing import Any, Callable, List, Optional, Type

from mava.callbacks import Callback


class Component(Callback):
    @abc.abstractmethod
    def __init__(self, config: Any) -> None:
        """By default, just set the local config.

        Args:
            config: Config for component, will be empty if no config class.
        """
        self.config = config

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """Static method that returns component name."""
        raise NotImplementedError("Name method not implemented for a component")

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        Returns:
            List of required component classes.
        """
        return []
