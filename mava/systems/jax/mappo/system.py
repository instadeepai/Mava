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

from mava.components.jax import building, executing, training, updating
from mava.specs import DesignSpec
from mava.systems.jax import System
from mava.systems.jax.mappo.components import ExtrasLogProbSpec
from mava.systems.jax.mappo.config import MAPPODefaultConfig


class MAPPOSystem(System):
    def design(self) -> Tuple[DesignSpec, Any]:
        """Mock system design with zero components.

        Returns:
            system callback components
        """
        # Set the default configs
        default_params = MAPPODefaultConfig()

        # Default system processes
        # System initialization
        system_init = DesignSpec(
            environment_spec=building.EnvironmentSpec, system_init=building.SystemInit
        ).get()

        # Executor
        executor_process = DesignSpec(
            executor_init=executing.ExecutorInit,
            executor_observe=executing.FeedforwardExecutorObserve,
            executor_select_action=executing.FeedforwardExecutorSelectAction,
            executor_adder=building.ParallelSequenceAdder,
            executor_environment_loop=building.ParallelExecutorEnvironmentLoop,
            networks=building.DefaultNetworks,
        ).get()

        # Trainer
        trainer_process = DesignSpec(
            trainer_init=training.TrainerInit,
            gae_fn=training.GAE,
            loss=training.MAPGWithTrustRegionClippingLoss,
            epoch_update=training.MAPGEpochUpdate,
            minibatch_update=training.MAPGMinibatchUpdate,
            sgd_step=training.MAPGWithTrustRegionStep,
            step=training.DefaultStep,
            trainer_dataset=building.TrajectoryDataset,
        ).get()

        # Data Server
        data_server_process = DesignSpec(
            data_server=building.OnPolicyDataServer,
            data_server_adder_signature=building.ParallelSequenceAdderSignature,
            extras_spec=ExtrasLogProbSpec,
        ).get()

        # Parameter Server
        parameter_server_process = DesignSpec(
            parameter_server=updating.DefaultParameterServer,
            executor_parameter_client=building.ExecutorParameterClient,
            trainer_parameter_client=building.TrainerParameterClient,
        ).get()

        system = DesignSpec(
            **system_init,
            **data_server_process,
            **parameter_server_process,
            **executor_process,
            **trainer_process,
            distributor=building.Distributor,
            logger=building.Logger,
        )
        return system, default_params
