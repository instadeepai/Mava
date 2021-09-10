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

"""Utilty Enums."""
from enum import Enum


class ArchitectureType(Enum):
    feedforward = 1
    recurrent = 2


class Network(Enum):
    mlp = 1
    atari_dqn_network = 2
    coms_network = 3


class Trainer(Enum):
    single_trainer = 1
    """Only create one trainer that trains all the networks."""
    one_trainer_per_network = 2
    """Create one trainer per network. Therefore each trainer is
    dedicated to training only one network."""


class NetworkSampler(Enum):
    fixed_agent_networks = 1
    """This option keeps the network used by each agent fixed."""
    random_agent_networks = 2
    """Create N network policies, where N is the number of agents. Randomly
    select policies from this sets for each agent at the start of a
    episode. This sampling is done with replacement so the same policy
    can be selected for more than one agent for a given episode."""
