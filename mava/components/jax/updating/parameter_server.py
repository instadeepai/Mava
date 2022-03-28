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

from mava.callbacks import Callback
from mava.core_jax import SystemBuilder, SystemParameterServer


@dataclass
class ParameterServerConfig:
    parameter_server_param: str = "Testing"
    Second_var: str = "Testing2"


class DefaultParameterServer(Callback):
    def __init__(
        self,
        config: ParameterServerConfig = ParameterServerConfig(),
    ) -> None:
        """Mock system component."""
        self.config = config

    def on_building_parameter_server_start(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        pass
        # networks = builder.attr.network_factory(
        #     environment_spec=builder.attr.environment_spec,
        #     agent_net_keys=builder.attr.agent_net_keys,
        # )

        # # Create parameters
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

        # parameters["trainer_steps"] = jnp.int32(0)
        # parameters["trainer_walltime"] = jnp.int32(0)
        # parameters["evaluator_steps"] = jnp.int32(0)
        # parameters["evaluator_episodes"] = jnp.int32(0)
        # parameters["executor_episodes"] = jnp.int32(0)
        # parameters["executor_steps"] = jnp.int32(0)

    # Get
    def on_parameter_server_get_parameters(self, server: SystemParameterServer) -> None:
        """_summary_"""
        pass

    # Set
    def on_parameter_server_set_parameters(self, server: SystemParameterServer) -> None:
        """_summary_"""
        pass

    # Add
    def on_parameter_server_add_to_parameters(
        self, server: SystemParameterServer
    ) -> None:
        """_summary_"""
        pass

    @property
    def name(self) -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "parameter_server"
