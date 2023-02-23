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

"""Jax IDQN system."""
from typing import Any, Tuple

from mava.components import building, executing, training, updating
from mava.components.building.adders import ParallelTransitionAdder
from mava.components.building.guardrails import ComponentDependencyGuardrails
from mava.specs import DesignSpec
from mava.systems import System
from mava.systems.idqn.components import building as dqn_building
from mava.systems.idqn.components import executing as dqn_executing
from mava.systems.idqn.components import training as dqn_training
from mava.systems.idqn.components.building.extras_spec import DQNExtrasSpec
from mava.systems.idqn.config import IDQNDefaultConfig


class IDQNSystem(System):
    @staticmethod
    def design() -> Tuple[DesignSpec, Any]:
        """System design for IPPO with single optimiser.

        Args:
            None.

        Returns:
            system: design spec for IPPO
            default_params: default IPPO configuration
        """
        # Set the default configs
        default_params = IDQNDefaultConfig()

        # Default system processes
        # System initialization
        system_init = DesignSpec(
            environment_spec=building.EnvironmentSpec,
            system_init=building.FixedNetworkSystemInit,
        ).get()

        # Executor
        executor_process = DesignSpec(
            executor_init=executing.ExecutorInit,
            executor_observe=dqn_executing.DQNFeedforwardExecutorObserve,
            executor_select_action=dqn_executing.DQNFeedforwardExecutorSelectAction,
            executor_adder=ParallelTransitionAdder,
            adder_priority=building.UniformAdderPriority,
            rate_limiter=building.reverb_components.SampleToInsertRateLimiter,
            executor_environment_loop=building.ParallelExecutorEnvironmentLoop,
            networks=building.DefaultNetworks,
            epsilon_scheduler=dqn_executing.EpsilonScheduler,
        ).get()

        # Trainer
        trainer_process = DesignSpec(
            optimisers=dqn_building.Optimiser,
            trainer_init=training.SingleTrainerInit,
            loss=dqn_training.IDQNLoss,
            sgd_step=dqn_training.IDQNStep,
            step=training.DefaultTrainerStep,
            trainer_dataset=building.TransitionDataset,
        ).get()

        # Data Server
        data_server_process = DesignSpec(
            data_server=building.OffPolicyDataServer,
            data_server_adder_signature=building.ParallelTransitionAdderSignature,
            data_server_remover=building.reverb_components.FIFORemover,
            data_server_sampler=building.reverb_components.PrioritySampler,
            extras_spec=DQNExtrasSpec,
        ).get()

        # Parameter Server
        parameter_server_process = DesignSpec(
            parameter_server=updating.DefaultParameterServer,
            executor_parameter_client=building.ExecutorParameterClient,
            trainer_parameter_client=building.TrainerParameterClient,
            termination_condition=updating.CountConditionTerminator,
            checkpointer=updating.Checkpointer,
        ).get()

        system = DesignSpec(
            **system_init,
            **data_server_process,
            **parameter_server_process,
            **executor_process,
            **trainer_process,
            distributor=building.Distributor,
            logger=building.Logger,
            component_dependency_guardrails=ComponentDependencyGuardrails,
        )
        return system, default_params
