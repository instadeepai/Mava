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

"""Builder components for Mava systems."""

from mava.components.building import reverb_components
from mava.components.building.adders import (
    ParallelSequenceAdder,
    ParallelSequenceAdderSignature,
    ParallelTransitionAdder,
    ParallelTransitionAdderSignature,
    UniformAdderPriority,
)
from mava.components.building.best_checkpointer import BestCheckpointer
from mava.components.building.data_server import OffPolicyDataServer, OnPolicyDataServer
from mava.components.building.datasets import TrajectoryDataset, TransitionDataset
from mava.components.building.distributor import Distributor
from mava.components.building.environments import (
    EnvironmentSpec,
    ParallelExecutorEnvironmentLoop,
)
from mava.components.building.extras_spec import ExtrasSpec
from mava.components.building.loggers import Logger
from mava.components.building.networks import DefaultNetworks
from mava.components.building.optimisers import DefaultOptimisers

# ActorCriticExecutorParameterClient,
# ActorCriticTrainerParameterClient,
from mava.components.building.parameter_client import (
    ExecutorParameterClient,
    TrainerParameterClient,
)
from mava.components.building.system_init import (
    CustomSamplingSystemInit,
    FixedNetworkSystemInit,
    RandomSamplingSystemInit,
)
