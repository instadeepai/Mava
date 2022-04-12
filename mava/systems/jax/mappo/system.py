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

"""Jax MAPPO system."""
from typing import Any, Tuple

from mava.components.jax.building.adders import (
    ParallelSequenceAdder,
    ParallelSequenceAdderSignature,
)
from mava.components.jax.building.data_server import OnPolicyDataServer
from mava.components.jax.building.datasets import TrajectoryDataset
from mava.components.jax.building.distributor import Distributor
from mava.components.jax.building.environments import ParallelExecutorEnvironmentLoop
from mava.components.jax.building.loggers import Logger
from mava.components.jax.building.networks import DefaultNetworks
from mava.components.jax.building.parameter_client import (
    ExecutorParameterClient,
    TrainerParameterClient,
)
from mava.components.jax.updating.parameter_server import DefaultParameterServer
from mava.specs import DesignSpec
from mava.systems.jax import System
from mava.systems.jax.mappo.components import ExtrasLogProbSpec
from mava.systems.jax.mappo.config import MAPPODefaultConfig
from mava.systems.jax.mappo.execution import EXECUTOR_SPEC
from mava.systems.jax.mappo.training import TRAINER_SPEC


class PPOSystem(System):
    def design(self) -> Tuple[DesignSpec, Any]:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        # Set the default configs
        default_params = MAPPODefaultConfig()

        # Default components
        executor = EXECUTOR_SPEC.get()
        executor_process = DesignSpec(
            **executor,
            executor_adder=ParallelSequenceAdder,
            executor_environment_loop=ParallelExecutorEnvironmentLoop,
            networks=DefaultNetworks,
        ).get()

        trainer = TRAINER_SPEC.get()
        trainer_process = DesignSpec(
            **trainer,
            trainer_dataset=TrajectoryDataset,
        ).get()

        data_server_process = DesignSpec(
            data_server=OnPolicyDataServer,
            data_server_adder_signature=ParallelSequenceAdderSignature,
            extras_spec=ExtrasLogProbSpec,
        ).get()

        parameter_server_process = DesignSpec(
            parameter_server=DefaultParameterServer,
            executor_parameter_client=ExecutorParameterClient,
            trainer_parameter_client=TrainerParameterClient,
        ).get()

        system = DesignSpec(
            **data_server_process,
            **parameter_server_process,
            **executor_process,
            **trainer_process,
            distributor=Distributor,
            logger=Logger,
        )
        return system, default_params
