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

"""Parameter server component for Mava systems."""
from dataclasses import dataclass

import numpy as np

from mava.callbacks import Callback
from mava.core_jax import SystemParameterServer


@dataclass
class ParameterServerConfig:
    pass


class DefaultParameterServer(Callback):
    def __init__(
        self,
        config: ParameterServerConfig = ParameterServerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_parameter_server_init_start(self, server: SystemParameterServer) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        # networks = builder.attr.network_factory(
        #     environment_spec=builder.attr.environment_spec,
        #     agent_net_keys=builder.attr.agent_net_keys,
        # )

        # Create parameters
        server.config.parameters = {
            "trainer_steps": np.zeros(1, dtype=np.int32),
            "trainer_walltime": np.zeros(1, dtype=np.float32),
            "evaluator_steps": np.zeros(1, dtype=np.int32),
            "evaluator_episodes": np.zeros(1, dtype=np.int32),
            "executor_episodes": np.zeros(1, dtype=np.int32),
            "executor_steps": np.zeros(1, dtype=np.int32),
        }

        # parameters = {}
        # rng_key = jax.random.PRNGKey(42)
        # # Network parameters
        # for net_type_key in networks.keys():
        #     for net_key in networks[net_type_key].keys():
        #         # Ensure obs and target networks are sonnet modules

        #         parameters[f"{net_key}_{net_type_key}"] = networks[net_type_key][
        #             net_key
        #         ].init(rng_key)

        #         rng_key, subkey = jax.random.split(rng_key)
        #         del subkey
        

    # Get
    def on_parameter_server_get_parameters(self, server: SystemParameterServer) -> None:
        """_summary_"""
        names = server.config._param_names

        if type(names) == str:
            get_params = server.config.parameters[names]  # type: ignore
        else:
            get_params = {}
            for var_key in names:
                get_params[var_key] = server.config.parameters[var_key]
        server.config.get_parameters = get_params

    # Set
    def on_parameter_server_set_parameters(self, server: SystemParameterServer) -> None:
        """_summary_"""
        params = server.config._set_params
        names = params.keys()

        if type(names) == str:
            params = {names: params}  # type: ignore
            names = [names]  # type: ignore

        for var_key in names:
            assert var_key in server.config.parameters
            if type(server.config.parameters[var_key]) == tuple:
                raise NotImplementedError
                # # Loop through tuple
                # for var_i in range(len(server.config.parameters[var_key])):
                #     server.config.parameters[var_key][var_i].assign(params[var_key][var_i])
            else:
                server.config.parameters[var_key] = params[var_key]

    # Add
    def on_parameter_server_add_to_parameters(
        self, server: SystemParameterServer
    ) -> None:
        """_summary_"""
        params = server.config._add_to_params
        names = params.keys()

        if type(names) == str:
            params = {names: params}  # type: ignore
            names = [names]  # type: ignore

        for var_key in names:
            assert var_key in server.config.parameters
            server.config.parameters[var_key] += params[var_key]

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "parameter_server"
