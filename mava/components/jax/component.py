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
from types import SimpleNamespace
from typing import Any, Callable, Optional

from mava.callbacks import Callback


class Component(Callback):
    @abc.abstractmethod
    def __init__(self, local_config: SimpleNamespace, global_config: SimpleNamespace) -> None:
        """_summary_

        Args:
            local_config : Config for this specific component, with type "config_class()".
            global_config : Namespace containing config for all components.
        """
        self.local_config = local_config
        self.global_config = global_config

    @staticmethod
    @abc.abstractmethod
    def name() -> str:
        """Static method that returns component name."""
        raise NotImplementedError("Name method not implemented for component")

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Optional class which specifies the dataclass/config object for the component.

        Returns:
            config class/dataclass for component.
        """
        pass
