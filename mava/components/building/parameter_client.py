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

"""Parameter client for system builders"""
import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Type

import numpy as np

from mava.callbacks import Callback
from mava.components import Component
from mava.components.training.trainer import BaseTrainerInit
from mava.core_jax import SystemBuilder
from mava.systems import ParameterClient


class BaseParameterClient(Component):
    def _set_up_count_parameters(
        self, params: Dict[str, Any]
    ) -> Tuple[List[str], Dict[str, Any]]:
        """Registers parameters to count and store.

        Counts trainer_steps, trainer_walltime, evaluator_steps,
        evaluator_episodes, executor_episodes, executor_steps.

        Args:
            params: Network parameters.

        Returns:
            Tuple of count parameters and network parameters.
        """
        add_params = {
            "trainer_steps": np.array(0, dtype=np.int32),
            "trainer_walltime": np.array(0, dtype=np.float32),
            "evaluator_steps": np.array(0, dtype=np.int32),
            "evaluator_episodes": np.array(0, dtype=np.int32),
            "executor_episodes": np.array(0, dtype=np.int32),
            "executor_steps": np.array(0, dtype=np.int32),
        }
        params.update(add_params)
        return list(add_params.keys()), params

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        BaseTrainerInit required to set up builder.store.networks
        and builder.store.trainer_networks.

        Returns:
            List of required component classes.
        """
        return [BaseTrainerInit]


@dataclass
class ExecutorParameterClientConfig:
    executor_parameter_update_period: int = 200


class ExecutorParameterClient(BaseParameterClient):
    def __init__(
        self,
        config: ExecutorParameterClientConfig = ExecutorParameterClientConfig(),
    ) -> None:
        """Component creates a parameter client for the executor.

        Args:
            config: ExecutorParameterClientConfig.
        """

        self.config = config

    def on_building_executor_parameter_client(self, builder: SystemBuilder) -> None:
        """Create and store the executor parameter client.

        Gets network parameters from store and registers them for tracking.
        Also works for the evaluator.

        Args:
            builder: SystemBuilder.
        """
        # Create policy parameters
        params: Dict[str, Any] = {}
        # Executor does not explicitly set variables i.e. it adds to count variables
        # and hence set_keys is empty
        set_keys: List[str] = []
        get_keys: List[str] = []

        for agent_net_key in builder.store.networks.keys():
            policy_param_key = f"policy_network-{agent_net_key}"
            params[policy_param_key] = builder.store.networks[
                agent_net_key
            ].policy_params
            get_keys.append(policy_param_key)

            critic_param_key = f"critic_network-{agent_net_key}"
            params[critic_param_key] = builder.store.networks[
                agent_net_key
            ].critic_params
            get_keys.append(critic_param_key)

        # Create observations' normalisation parameters
        params["norm_params"] = builder.store.norm_params
        get_keys.append("norm_params")

        # Create best performance network params in case of evaluator
        if builder.store.is_evaluator and builder.store.checkpoint_best_perf:
            builder.store.best_checkpoint: Dict[str, Any] = {}
            for metric in builder.store.metrics_checkpoint:
                builder.store.best_checkpoint[metric] = {}
                builder.store.best_checkpoint[metric]["best_performance"] = None
                for agent_net_key in builder.store.networks.keys():
                    builder.store.best_checkpoint[metric][
                        f"policy_network-{agent_net_key}"
                    ] = copy.deepcopy(
                        builder.store.networks[agent_net_key].policy_params
                    )
                    builder.store.best_checkpoint[metric][
                        f"critic_network-{agent_net_key}"
                    ] = copy.deepcopy(
                        builder.store.networks[agent_net_key].critic_params
                    )
                    builder.store.best_checkpoint[metric][
                        f"policy_opt_state-{agent_net_key}"
                    ] = copy.deepcopy(builder.store.policy_opt_states[agent_net_key])
                    builder.store.best_checkpoint[metric][
                        f"critic_opt_state-{agent_net_key}"
                    ] = copy.deepcopy(builder.store.critic_opt_states[agent_net_key])
            params["best_checkpoint"] = builder.store.best_checkpoint
            set_keys.append("best_checkpoint")

        count_names, params = self._set_up_count_parameters(params=params)

        get_keys.extend(count_names)

        builder.store.executor_counts = {name: params[name] for name in count_names}

        parameter_client = None
        if builder.store.parameter_server_client:
            # Create parameter client
            parameter_client = ParameterClient(
                client=builder.store.parameter_server_client,
                parameters=params,
                get_keys=get_keys,
                set_keys=set_keys,
                update_period=self.config.executor_parameter_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning parameters before running the environment loop.
            parameter_client.get_and_wait()

        builder.store.executor_parameter_client = parameter_client

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "executor_parameter_client"


@dataclass
class TrainerParameterClientConfig:
    trainer_parameter_update_period: int = 5


class TrainerParameterClient(BaseParameterClient):
    def __init__(
        self,
        config: TrainerParameterClientConfig = TrainerParameterClientConfig(),
    ) -> None:
        """Component creates a parameter client for the trainer.

        Args:
            config: TrainerParameterClientConfig.
        """

        self.config = config

    def on_building_trainer_parameter_client(self, builder: SystemBuilder) -> None:
        """Create and store the trainer parameter client.

        Gets network parameters from store and registers them for tracking.

        Args:
            builder: SystemBuilder.
        """
        # Create parameter client
        params: Dict[str, Any] = {}
        set_keys: List[str] = []
        get_keys: List[str] = []
        # TODO (dries): Only add the networks this trainer is working with.
        # Not all of them.
        trainer_networks = builder.store.trainer_networks[builder.store.trainer_id]

        for net_key in builder.store.networks.keys():
            params[f"policy_network-{net_key}"] = builder.store.networks[
                net_key
            ].policy_params
            params[f"critic_network-{net_key}"] = builder.store.networks[
                net_key
            ].critic_params

            if net_key in set(trainer_networks):
                set_keys.append(f"policy_network-{net_key}")
                set_keys.append(f"critic_network-{net_key}")
            else:
                get_keys.append(f"policy_network-{net_key}")
                get_keys.append(f"critic_network-{net_key}")

            params[f"policy_opt_state-{net_key}"] = builder.store.policy_opt_states[
                net_key
            ]
            params[f"critic_opt_state-{net_key}"] = builder.store.critic_opt_states[
                net_key
            ]
            set_keys.append(f"policy_opt_state-{net_key}")
            set_keys.append(f"critic_opt_state-{net_key}")

        # Add observations' normalisation parameters
        params["norm_params"] = builder.store.norm_params
        set_keys.append("norm_params")

        count_names, params = self._set_up_count_parameters(params=params)

        get_keys.extend(count_names)
        builder.store.trainer_counts = {name: params[name] for name in count_names}

        # Create parameter client
        parameter_client = None
        if builder.store.parameter_server_client:
            parameter_client = ParameterClient(
                client=builder.store.parameter_server_client,
                parameters=params,
                get_keys=get_keys,
                set_keys=set_keys,
                update_period=self.config.trainer_parameter_update_period,
            )

            # Get all the initial parameters
            parameter_client.get_all_and_wait()

        builder.store.trainer_parameter_client = parameter_client

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "trainer_parameter_client"
