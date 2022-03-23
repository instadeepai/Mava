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
from dataclasses import dataclass

from mava.callbacks import Callback


@dataclass
class Config:
    pass


class Component(Callback):
    @abc.abstractmethod
    def __init__(self, config: Config = Config()):
        """_summary_

        Args:
            config : _description_.
        """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """_summary_"""
