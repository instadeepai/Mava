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
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence, Type, Union

import numpy as np
from acme.jax import savers

from mava.callbacks import Callback
from mava.components.jax.building.networks import Networks
from mava.components.jax.component import Component
from mava.core_jax import SystemParameterServer


@dataclass
class ParameterServerConfig:
    checkpoint: bool = True
    experiment_path: str = "~/mava/"
    checkpoint_minute_interval: int = 5
    non_blocking_sleep_seconds: int = 10


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
        """Register parameters and network params to track. Create checkpointer."""
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

    # Save variables using checkpointer
    @abc.abstractmethod
    def on_parameter_server_run_loop(self, server: SystemParameterServer) -> None:
        """Intermittently checkpoint the server parameters."""
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
        Creates the checkpointer.
        Handles the getting, setting, and adding of parameters.
        Handles periodic checkpointing of parameters.

        Args:
            config: ParameterServerConfig.
        """
        self.config = config

    def on_parameter_server_init_start(self, server: SystemParameterServer) -> None:
        """Register parameters and network params to track. Create checkpointer.

        Args:
            server: SystemParameterServer.
        """
        networks = server.store.network_factory()

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
        for net_type_key in networks.keys():
            for agent_net_key in networks[net_type_key].keys():
                # Ensure obs and target networks are sonnet modules
                server.store.parameters[
                    f"policy_{net_type_key}-{agent_net_key}"
                ] = networks[net_type_key][agent_net_key].policy_params

                # Ensure obs and target networks are sonnet modules
                server.store.parameters[
                    f"critic_{net_type_key}-{agent_net_key}"
                ] = networks[net_type_key][agent_net_key].critic_params

        # Create the checkpointer
        if self.config.checkpoint:
            server.store.last_checkpoint_time = 0

            # Only save variables that are not empty.
            save_variables = {}
            for key in server.store.parameters.keys():
                var = server.store.parameters[key]
                # Don't store empty tuple (e.g. empty observation_network) variables
                if not (type(var) == tuple and len(var) == 0):
                    save_variables[key] = var
            server.store.system_checkpointer = savers.Checkpointer(
                save_variables, self.config.experiment_path, time_delta_minutes=0
            )

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

    # Save variables using checkpointer
    def on_parameter_server_run_loop(self, server: SystemParameterServer) -> None:
        """Intermittently checkpoint the server parameters.

        Args:
            server: SystemParameterServer.

        Returns:
            None.
        """
        if (
            self.config.checkpoint
            and server.store.last_checkpoint_time
            + self.config.checkpoint_minute_interval * 60
            + 1
            < time.time()
        ):
            server.store.system_checkpointer.save()
            server.store.last_checkpoint_time = time.time()
            print("Updated variables checkpoint.")
