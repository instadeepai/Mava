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
from typing import Dict

import sonnet as snt

"""Base architecture interface for multi-agent RL systems"""


class BaseArchitecture:
    """Base class for MARL architectures.
    Objects which implement this interface provide a set of functions
    to create systems according to a specific architectural design,
    e.g. decentralised, centralised or networked.
    """

    @abc.abstractmethod
    def create_actor_variables(self) -> Dict[str, Dict[str, snt.Module]]:
        """Create network variables for actors in the system."""

    @abc.abstractmethod
    def create_system(self) -> Dict[str, Dict[str, snt.Module]]:
        """Create system architecture."""


class BasePolicyArchitecture(BaseArchitecture):
    """Base class for policy gradient MARL architectures.
    Objects which implement this interface provide a set of functions
    to create systems according to a specific architectural design,
    e.g. decentralised, centralised or networked.
    """

    @abc.abstractmethod
    def create_behaviour_policy(self) -> Dict[str, Dict[str, snt.Module]]:
        """Return behaviour policy networks (observation network + policy head)."""


class BaseActorCritic(BasePolicyArchitecture):
    """Base class for MARL Actor critic architectures.
    Objects which implement this interface provide a set of functions
    to create systems according to a specific architectural design,
    e.g. decentralised, centralised or networked.
    """

    @abc.abstractmethod
    def create_critic_variables(self) -> Dict[str, Dict[str, snt.Module]]:
        """Create network variables for critics in the system."""
