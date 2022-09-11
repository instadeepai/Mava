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

"""Commonly used replay table components for system builders"""
import abc
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type

import reverb

from mava import specs
from mava.callbacks import Callback
from mava.components.jax import Component
from mava.components.jax.building.adders import AdderSignature
from mava.components.jax.building.environments import EnvironmentSpec
from mava.components.jax.building.reverb_components import RateLimiter, Remover, Sampler
from mava.components.jax.building.system_init import BaseSystemInit
from mava.core_jax import SystemBuilder
from mava.utils import enums
from mava.utils.builder_utils import convert_specs
from mava.utils.sort_utils import sort_str_num


class DataServer(Component):
    def __init__(
        self,
        config: Any,
    ) -> None:
        """Component sets up a reverb Table for each trainer.

        Args:
            config: Any.
        """

        self.config = config

    def _create_table_per_trainer(self, builder: SystemBuilder) -> List[reverb.Table]:
        """Create a reverb table for each trainer.

        Defines a default table network config for when one is not provided, but
        which only works when using fixed agent networks.
        Converts specs from the environment and uses them to create tables.

        Args:
            builder: SystemBuilder.

        Returns:
            List of reverb tables where each table corresponds to a trainer.
        """
        data_tables = []
        # Default table network config - often overwritten by TrainerInit.
        if not hasattr(builder.store, "table_network_config"):
            builder.store.table_network_config = {
                "table_0": sort_str_num(builder.store.agent_net_keys.values())
            }
            assert (
                builder.store.global_config.network_sampling_setup
                == enums.NetworkSampler.fixed_agent_networks
            ), f"We only have a default config for the fixed_agent_networks sampler setting, \
            not the {builder.store.global_config.network_sampling_setup} setting."

        for table_key in builder.store.table_network_config.keys():
            # TODO (dries): Clean the below converter code up.
            # Convert a Mava spec
            num_networks = len(builder.store.table_network_config[table_key])
            env_specs = copy.deepcopy(builder.store.ma_environment_spec)
            env_specs.set_agent_environment_specs(
                convert_specs(
                    builder.store.agent_net_keys,
                    env_specs.get_agent_environment_specs(),
                    num_networks,
                )
            )

            env_specs._keys = list(
                sort_str_num(env_specs.get_agent_environment_specs().keys())
            )
            if env_specs.get_extras_specs() is not None:
                env_specs.set_extras_specs(
                    convert_specs(
                        builder.store.agent_net_keys,
                        env_specs.get_extras_specs(),
                        num_networks,
                    )
                )
            extras_specs = convert_specs(
                builder.store.agent_net_keys,
                builder.store.extras_spec,
                num_networks,
            )
            table = self.table(table_key, env_specs, extras_specs, builder)
            data_tables.append(table)
        return data_tables

    @abc.abstractmethod
    def table(
        self,
        table_key: str,
        environment_specs: specs.MAEnvironmentSpec,
        extras_specs: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """Abstract method defining signature for table creation.

        Args:
            table_key: Identifier for table.
            environment_specs: Environment specs.
            extras_specs: Other specs.
            builder: SystemBuilder.

        Returns:
            A new reverb table.
        """

    def on_building_data_server(self, builder: SystemBuilder) -> None:
        """Create a table for each trainer and load into store.

        Args:
            builder: SystemBuilder

        Returns:
            None.
        """
        builder.store.data_tables = self._create_table_per_trainer(builder)

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "data_server"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        BaseSystemInit required to set up builder.store.agent_net_keys
        and for config network_sampling_setup.
        EnvironmentSpec required to set up builder.store.environment_spec
        and builder.store.extras_spec.
        AdderSignature required to set up builder.store.adder_signature_fn.

        Returns:
            List of required component classes.
        """
        return [EnvironmentSpec, AdderSignature, BaseSystemInit]


@dataclass
class OffPolicyDataServerConfig:
    max_size: int = 100000
    max_times_sampled: int = 0


class OffPolicyDataServer(DataServer):
    def __init__(
        self, config: OffPolicyDataServerConfig = OffPolicyDataServerConfig()
    ) -> None:
        """Component creates an off-policy data server.

        Args:
            config: OffPolicyDataServerConfig.
        """

        self.config = config

    def table(
        self,
        table_key: str,
        environment_specs: specs.MAEnvironmentSpec,
        extras_specs: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """Create OffPolicyDataServer table.

        Requires sampler and remover functions in the system to operate.

        Args:
            table_key: Identifier for table.
            environment_specs: Environment specs.
            extras_specs: Other specs.
            builder: SystemBuilder.

        Returns:
            A new reverb table.
        """
        if not hasattr(builder.store, "sampler_fn"):
            raise ValueError(
                "A sampler component for the dataserver has not been given"
            )

        if not hasattr(builder.store, "remover_fn"):
            raise ValueError(
                "A remover component for the dataserver has not been given"
            )

        table = reverb.Table(
            name=table_key,
            sampler=builder.store.sampler_fn(),
            remover=builder.store.remover_fn(),
            max_size=self.config.max_size,
            rate_limiter=builder.store.rate_limiter_fn(),
            signature=builder.store.adder_signature_fn(environment_specs, extras_specs),
            max_times_sampled=self.config.max_times_sampled,
        )
        return table


    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        RateLimiter required to set up builder.store.rate_limiter_fn.
        Remover required to set up builder.store.remover_fn.
        Sampler required to set up builder.store.sampler_fn.

        Returns:
            List of required component classes.
        """
        return DataServer.required_components() + [RateLimiter, Remover, Sampler]


@dataclass
class OnPolicyDataServerConfig:
    max_queue_size: int = 1000


class OnPolicyDataServer(DataServer):
    def __init__(
        self,
        config: OnPolicyDataServerConfig = OnPolicyDataServerConfig(),
    ) -> None:
        """Component creates an on-policy data server.

        Args:
            config: OnPolicyDataServerConfig.
        """

        self.config = config

    def table(
        self,
        table_key: str,
        environment_specs: specs.MAEnvironmentSpec,
        extras_specs: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """Create OnPolicyDataServer table.

        Requires sampler and remover functions in the system to operate.

        Args:
            table_key: Identifier for table.
            environment_specs: Environment specs.
            extras_specs: Other specs.
            builder: SystemBuilder.

        Returns:
            A new reverb table.
        """
        if hasattr(builder.store.global_config, "sequence_length"):
            signature = builder.store.adder_signature_fn(
                environment_specs,
                builder.store.global_config.sequence_length,
                extras_specs,
            )
        else:
            signature = builder.store.adder_signature_fn(
                environment_specs, extras_specs
            )
        table = reverb.Table.queue(
            name=table_key,
            max_size=self.config.max_queue_size,
            signature=signature,
        )
        return table
