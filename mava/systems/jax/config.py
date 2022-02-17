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

"""Config class for Mava systems"""

from dataclasses import is_dataclass
from types import SimpleNamespace
from typing import Any, Dict

from mava.utils.config_utils import flatten_dict


class Config:
    """_summary_"""

    def __init__(self) -> None:
        """_summary_"""
        self._config: Dict = {}
        self._built = False

    def add(self, **kwargs: type) -> None:
        """_summary_

        Raises:
            Exception: _description_
            Exception: _description_
        """
        for name, dataclass in kwargs.items():
            if is_dataclass(dataclass):
                if name in list(self._config.keys()):
                    raise Exception(
                        "The given component config is already part of the current \
                        system. Perhaps try updating it instead using .update()."
                    )
                else:
                    self._config[name] = dataclass
            else:
                raise Exception("Component configs must be a dataclass.")

    def build(self) -> None:
        """_summary_

        Returns:
            _description_
        """
        config_unwrapped: Dict = {}
        for param in self._config.values():
            config_unwrapped.update(flatten_dict(param.__dict__))
        self._config = config_unwrapped
        self._built = True

    def update(self, **kwargs: Any) -> None:
        """_summary_

        Raises:
            Exception: _description_
        """

        if not self._built:
            raise Exception(
                "Config must first be built using .build() before it can \
                be updated."
            )
        for name, param_value in kwargs.items():
            if name in list(self._config.keys()):
                self._config[name] = param_value
            else:
                raise Exception(
                    "The given parameter is not part of the current system. \
                    This should have been added first via a component .add() \
                    during system building."
                )

    def get(self) -> SimpleNamespace:
        """_summary_

        Returns:
            _description_
        """
        return SimpleNamespace(**self._config)
