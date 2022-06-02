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
import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from acme.jax import savers

from mava.components.jax.component import Component
from mava.core_jax import SystemParameterServer


@dataclass
class ParameterServerConfig:
    checkpoint: bool = True
    checkpoint_subpath: str = "~/mava/"
    checkpoint_minute_interval: int = 5
    non_blocking_sleep_seconds: int = 10


class DefaultParameterServer(Component):
    def __init__(
        self,
        config: ParameterServerConfig = ParameterServerConfig(),
    ) -> None:
        """Mock system Component."""
        self.config = config

    def on_parameter_server_init_start(self, server: SystemParameterServer) -> None:
        """_summary_

        Args:
            server : _description_
        """

        server.store.non_blocking_sleep_seconds = self.config.non_blocking_sleep_seconds
        networks = server.store.network_factory()

        # # Create parameters
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
                server.store.parameters[f"{net_type_key}-{agent_net_key}"] = networks[
                    net_type_key
                ][agent_net_key].params

        # Create the checkpointer
        if self.config.checkpoint:
            server.store.last_checkpoint_time = 0
            server.store.checkpoint_minute_interval = (
                self.config.checkpoint_minute_interval
            )

            # Only save variables that are not empty.
            save_variables = {}
            for key in server.store.parameters.keys():
                var = server.store.parameters[key]
                # Don't store empty tuple (e.g. empty observation_network) variables
                if not (type(var) == tuple and len(var) == 0):
                    save_variables[key] = var
            server.store.system_checkpointer = savers.Checkpointer(
                save_variables, self.config.checkpoint_subpath, time_delta_minutes=0
            )

    # Get
    def on_parameter_server_get_parameters(self, server: SystemParameterServer) -> None:
        """_summary_"""
        names = server.store._param_names

        if type(names) == str:
            get_params = server.store.parameters[names]  # type: ignore
        else:
            get_params = {}
            for var_key in names:
                get_params[var_key] = server.store.parameters[var_key]
        server.store.get_parameters = get_params

    # Set
    def on_parameter_server_set_parameters(self, server: SystemParameterServer) -> None:
        """_summary_"""
        params = server.store._set_params
        names = params.keys()

        if type(names) == str:
            params = {names: params}  # type: ignore
            names = [names]  # type: ignore

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
        """_summary_"""
        params = server.store._add_to_params
        names = params.keys()

        if type(names) == str:
            params = {names: params}  # type: ignore
            names = [names]  # type: ignore

        for var_key in names:
            assert var_key in server.store.parameters
            server.store.parameters[var_key] += params[var_key]

    # Save variables using checkpointer
    def on_parameter_server_run_loop(self, server: SystemParameterServer) -> None:
        """_summary_

        Args:
            server : _description_
        """
        if (
            server.store.system_checkpointer
            and server.store.last_checkpoint_time
            + server.store.checkpoint_minute_interval * 60
            + 1
            < time.time()
        ):
            server.store.system_checkpointer.save()
            server.store.last_checkpoint_time = time.time()
            print("Updated variables checkpoint.")

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "parameter_server"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for Component.

        Returns:
            config class/dataclass for Component.
        """
        return ParameterServerConfig
