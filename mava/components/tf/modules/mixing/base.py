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
from typing import Dict, Optional

import sonnet as snt

from mava import specs as mava_specs
from mava.components.tf.architectures import BaseArchitecture

"""Base mixing interface for multi-agent RL systems"""


class BaseMixingModule:
    """Base class for MARL mixing.
    Objects which implement this interface provide a set of functions
    to create systems that can perform value decomposition via a mixing
    strategy between agents in a multi-agent RL system.
    """

    @abc.abstractmethod
    def __init__(
        self,
        architecture: Optional[BaseArchitecture] = None,
        environment_spec: Optional[mava_specs.MAEnvironmentSpec] = None,
        agent_networks: Optional[Dict[str, snt.Module]] = None,
    ) -> None:
        """Initialise the mixer."""

    @abc.abstractmethod
    def _create_mixing_layer(self, name: str) -> snt.Module:
        """Abstract function for adding an arbitrary mixing layer to a
        given architecture."""

    @abc.abstractmethod
    def create_system(self) -> Dict[str, Dict[str, snt.Module]]:
        """Create/update system architecture with specified mixing."""
