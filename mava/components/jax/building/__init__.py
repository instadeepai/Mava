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

from mava.components.jax.building.adders import (  # noqa: E402
    ParallelSequenceAdder,
    ParallelSequenceAdderSignature,
    ParallelTransitionAdder,
    ParallelTransitionAdderSignature,
    UniformAdderPriority,
)
from mava.components.jax.building.data_server import (
    OffPolicyDataServer,
    OnPolicyDataServer,
)
from mava.components.jax.building.datasets import TrajectoryDataset, TransitionDataset
from mava.components.jax.building.distributor import Distributor
from mava.components.jax.building.environments import (
    EnvironmentSpec,
    ParallelExecutorEnvironmentLoop,
)
from mava.components.jax.building.loggers import Logger
from mava.components.jax.building.networks import DefaultNetworks
from mava.components.jax.building.parameter_client import (
    ExecutorParameterClient,
    ExecutorParameterClientSeparateNetworks,
    TrainerParameterClient,
    TrainerParameterClientSeparateNetworks,
)

# For off-policy learning
from mava.components.jax.building.rate_limiters import (
    MinSizeRateLimiter,
    SampleToInsertRateLimiter,
)
from mava.components.jax.building.reverb_components import Remover, Sampler
from mava.components.jax.building.system_init import (
    CustomSamplingSystemInit,
    FixedNetworkSystemInit,
    RandomSamplingSystemInit,
)
