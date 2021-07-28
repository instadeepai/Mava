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

import abc
from typing import Any, Union

from mava.systems.tf.mad4pg.system import MAD4PG
from mava.systems.tf.maddpg.system import MADDPG

"""Base communication interface for multi-agent RL systems"""
supported_pbt_systems = [MADDPG, MAD4PG]


class BasePBTModule:
    """Base class for PBT using a MARL system.
    Objects which implement this interface provide a set of functions
    to create systems that can perform some form of communication between
    agents in a multi-agent RL system.
    """

    @abc.abstractmethod
    def __init__(
        self,
        system: Union[MADDPG, MAD4PG],
    ) -> None:
        """Initializes the broadcaster communicator.
        Args:
            architecture: the BaseArchitecture used.
            shared: if a shared communication channel is used.
            channel_noise: stddev of normal noise in channel.
        """
        if type(system) not in supported_pbt_systems:
            raise NotImplementedError(
                f"Currently only {supported_pbt_systems} has "
                "the correct hooks to support PBT."
            )
        self._system = system

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._system, name)
