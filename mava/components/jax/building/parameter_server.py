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

"""Execution components for system builders"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp

from mava.components.jax import Component
from mava.core_jax import SystemBuilder


@dataclass
class ParameterServerProcessConfig:
    random_param: int = 5


class ParameterServerProcess(Component):
    def __init__(
        self, config: ParameterServerProcessConfig = ParameterServerProcessConfig()
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_parameter_server_start(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """
        networks = builder.attr.network_factory(
            environment_spec=builder.attr.environment_spec,
            agent_net_keys=builder.attr.agent_net_keys,
        )

        # Create parameters
        parameters = {}
        rng_key = jax.random.PRNGKey(42)
        # Network parameters
        for net_type_key in networks.keys():
            for net_key in networks[net_type_key].keys():
                # Ensure obs and target networks are sonnet modules

                parameters[f"{net_key}_{net_type_key}"] = networks[net_type_key][
                    net_key
                ].init(rng_key)

                rng_key, subkey = jax.random.split(rng_key)
                del subkey

        parameters["trainer_steps"] = jnp.int32(0)
        parameters["trainer_walltime"] = jnp.int32(0)
        parameters["evaluator_steps"] = jnp.int32(0)
        parameters["evaluator_episodes"] = jnp.int32(0)
        parameters["executor_episodes"] = jnp.int32(0)
        parameters["executor_steps"] = jnp.int32(0)

    def name(self) -> str:
        """_summary_"""
        return "parameter_server"
