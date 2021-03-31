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

# TODO (StJohn): complete base class for mixing

import abc
from typing import Dict

import sonnet as snt

"""Base mixing interface for multi-agent RL systems"""


class BaseMixingModule:
    """Base class for MARL mixing.
    Objects which implement this interface provide a set of functions
    to create systems that can perform value decomposition via a mixing
    strategy between agents in a multi-agent RL system.
    """

    @abc.abstractmethod
    def some_abstract_mixing_function(self) -> Dict[str, Dict[str, snt.Module]]:
        """Abstract communication function."""

    @abc.abstractmethod
    def create_system(self) -> Dict[str, Dict[str, snt.Module]]:
        """Create system architecture with mixing."""
