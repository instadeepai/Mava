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

"""Parameter server Component for Mava systems."""
import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Type, Union

import numpy as np

from mava.callbacks import Callback
from mava.components.building.networks import Networks
from mava.components.component import Component
from mava.core_jax import SystemParameterServer
from mava.utils.lp_utils import termination_fn


@dataclass
class ParameterServerConfig:
    non_blocking_sleep_seconds: int = 10
    experiment_path: str = "~/mava/"


class ParameterServer(Component):
    @abc.abstractmethod
    def __init__(
        self,
        config: ParameterServerConfig = ParameterServerConfig(),
    ) -> None:
        """Component defining hooks to override when creating a parameter server."""
        self.config = config

    @abc.abstractmethod
    def on_parameter_server_init_start(self, server: SystemParameterServer) -> None:
        """Register parameters and network params to track."""
        pass

    # Get
    @abc.abstractmethod
    def on_parameter_server_get_parameters(self, server: SystemParameterServer) -> None:
        """Fetch the parameters from the server specified in the store."""
        pass

    # Set
    @abc.abstractmethod
    def on_parameter_server_set_parameters(self, server: SystemParameterServer) -> None:
        """Set the parameters in the server to the values specified in the store."""
        pass

    # Add
    @abc.abstractmethod
    def on_parameter_server_add_to_parameters(
        self, server: SystemParameterServer
    ) -> None:
        """Increment the server parameters by the amount specified in the store."""
        pass

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "parameter_server"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        Networks required to set up server.store.network_factory.

        Returns:
            List of required component classes.
        """
        return [Networks]


class DefaultParameterServer(ParameterServer):
    def __init__(
        self,
        config: ParameterServerConfig = ParameterServerConfig(),
    ) -> None:
        """Default Mava parameter server.

        Registers count parameters and network params for tracking.
        Handles the getting, setting, and adding of parameters.

        Args:
            config: ParameterServerConfig.
        """
        self.config = config

    def on_parameter_server_init_start(self, server: SystemParameterServer) -> None:
        """Register parameters and network params to track.

        Args:
            server: SystemParameterServer.
        """
        networks = server.store.network_factory()

        # Store net_keys
        server.store.agents_net_keys = list(networks.keys())

        # Create parameters
        server.store.parameters = {
            "trainer_steps": np.zeros(1, dtype=np.int32),
            "trainer_walltime": np.zeros(1, dtype=np.float32),
            "evaluator_steps": np.zeros(1, dtype=np.int32),
            "evaluator_episodes": np.zeros(1, dtype=np.int32),
            "executor_episodes": np.zeros(1, dtype=np.int32),
            "executor_steps": np.zeros(1, dtype=np.int32),
        }
        # Network parameters
        for agent_net_key in networks.keys():
            # Ensure obs and target networks are sonnet modules
            server.store.parameters[f"policy_network-{agent_net_key}"] = networks[
                agent_net_key
            ].policy_params
            # Ensure obs and target networks are sonnet modules
            server.store.parameters[f"critic_network-{agent_net_key}"] = networks[
                agent_net_key
            ].critic_params
            server.store.parameters[
                f"policy_opt_state-{agent_net_key}"
            ] = server.store.policy_opt_states[agent_net_key]
            server.store.parameters[
                f"critic_opt_state-{agent_net_key}"
            ] = server.store.critic_opt_states[agent_net_key]

        # Normalization parameters
        server.store.parameters["norm_params"] = server.store.norm_params

        server.store.experiment_path = self.config.experiment_path

        # Interrupt the system in case evaluator or trainer failed
        server.store.parameters["evaluator_or_trainer_failed"] = False

        # Interrupt the system in case all the executors failed
        server.store.parameters["num_executor_failed"] = 0

        # Initiate best performance network values
        server.store.parameters["best_checkpoint"] = {}
        for metric in server.store.metrics_checkpoint:
            server.store.parameters["best_checkpoint"][metric] = {}
            for agent_net_key in networks.keys():
                server.store.parameters["best_checkpoint"][metric][
                    f"policy_network-{agent_net_key}"
                ] = networks[agent_net_key].policy_params
                server.store.parameters["best_checkpoint"][metric][
                    f"critic_network-{agent_net_key}"
                ] = networks[agent_net_key].critic_params
                server.store.parameters["best_checkpoint"][metric][
                    f"policy_opt_state-{agent_net_key}"
                ] = server.store.policy_opt_states[agent_net_key]
                server.store.parameters["best_checkpoint"][metric][
                    f"critic_opt_state-{agent_net_key}"
                ] = server.store.critic_opt_states[agent_net_key]

    # Get
    def on_parameter_server_get_parameters(self, server: SystemParameterServer) -> None:
        """Fetch the parameters from the server specified in the store.

        Args:
            server: SystemParameterServer.

        Returns:
            None.
        """
        # server.store._param_names set by Parameter Server
        names: Union[str, Sequence[str]] = server.store._param_names

        if type(names) == str:
            get_params = server.store.parameters[names]  # type: ignore
        else:
            get_params = {}
            for var_key in names:
                get_params[var_key] = server.store.parameters[var_key]
        server.store.get_parameters = get_params

        # Interrupt the system in case the evaluator or the trainer failed
        if server.store.parameters["evaluator_or_trainer_failed"]:
            termination_fn(server)

        # Interrupt the system in case all the executors failed
        if server.store.num_executors == server.store.parameters["num_executor_failed"]:
            termination_fn(server)

    # Set
    def on_parameter_server_set_parameters(self, server: SystemParameterServer) -> None:
        """Set the parameters in the server to the values specified in the store.

        Args:
            server: SystemParameterServer.

        Returns:
            None.
        """
        # server.store._set_params set by Parameter Server
        params: Dict[str, Any] = server.store._set_params
        names = params.keys()

        for var_key in names:
            assert var_key in server.store.parameters
            if type(server.store.parameters[var_key]) == tuple:
                raise NotImplementedError
                # # Loop through tuple
                # for var_i in range(len(server.store.parameters[var_key])):
                #     server.store.parameters[var_key][var_i].assign(params[var_key][var_i])
            elif type(server.store.parameters[var_key]) == dict:
                # In case we have 2 dimensional dict
                for name in params[var_key].keys():
                    server.store.parameters[var_key][name] = params[var_key][name]
            else:
                server.store.parameters[var_key] = params[var_key]

    # Add
    def on_parameter_server_add_to_parameters(
        self, server: SystemParameterServer
    ) -> None:
        """Increment the server parameters by the amount specified in the store.

        Args:
            server: SystemParameterServer.

        Returns:
            None.
        """
        # server.store._add_to_params set by Parameter Server
        params: Dict[str, Any] = server.store._add_to_params
        names = params.keys()

        for var_key in names:
            assert var_key in server.store.parameters
            server.store.parameters[var_key] += params[var_key]
