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
from reverb import rate_limiters, reverb_types

from mava import specs
from mava.callbacks import Callback
from mava.components.jax import Component
from mava.components.jax.building import EnvironmentSpec
from mava.components.jax.building.adders import AdderSignature
from mava.components.jax.building.rate_limiters import RateLimiter
from mava.components.jax.training import TrainerInit
from mava.core_jax import SystemBuilder
from mava.utils import enums
from mava.utils.builder_utils import covert_specs
from mava.utils.sort_utils import sort_str_num


class DataServer(Component):
    def __init__(
        self,
        config: Any,
    ) -> None:
        """_summary_

        Args:
            config : _description_.
        """

        self.config = config

    def _create_table_per_trainer(self, builder: SystemBuilder) -> List[reverb.Table]:
        data_tables = []
        # Default table network config - often overwritten by TrainerInit.
        if not hasattr(builder.store, "table_network_config"):
            builder.store.table_network_config = {
                "table_0": sort_str_num(builder.store.agent_net_keys.values())
            }
            assert (
                builder.store.global_config.network_sampling_setup_type
                == enums.NetworkSampler.fixed_agent_networks
            ), f"We only have a default config for the fixed_agent_networks sampler setting, \
            not the {builder.store.global_config.network_sampling_setup_type} setting."

        for table_key in builder.store.table_network_config.keys():
            # TODO (dries): Clean the below coverter code up.
            # Convert a Mava spec
            num_networks = len(builder.store.table_network_config[table_key])
            env_spec = copy.deepcopy(builder.store.environment_spec)
            env_spec._specs = covert_specs(
                builder.store.agent_net_keys, env_spec._specs, num_networks
            )

            env_spec._keys = list(sort_str_num(env_spec._specs.keys()))
            if env_spec.extra_specs is not None:
                env_spec.extra_specs = covert_specs(
                    builder.store.agent_net_keys, env_spec.extra_specs, num_networks
                )
            extras_spec = covert_specs(
                builder.store.agent_net_keys,
                builder.store.extras_spec,
                num_networks,
            )
            table = self.table(table_key, env_spec, extras_spec, builder)
            data_tables.append(table)
        return data_tables

    @abc.abstractmethod
    def table(
        self,
        table_key: str,
        environment_spec: specs.MAEnvironmentSpec,
        extras_spec: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """_summary_"""

    def on_building_data_server(self, builder: SystemBuilder) -> None:
        """[summary]"""
        builder.store.data_tables = self._create_table_per_trainer(builder)

    @staticmethod
    def name() -> str:
        """Component type name, e.g. 'dataset' or 'executor'."""
        return "data_server"

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        BaseSystemInit required to set up builder.store.agent_net_keys.
        EnvironmentSpec required to set up builder.store.environment_spec
        and builder.store.extras_spec.
        AdderSignature required to set up builder.store.adder_signature_fn.

        Returns:
            List of required component classes.
        """
        return [TrainerInit, EnvironmentSpec, AdderSignature]


@dataclass
class OffPolicyDataServerConfig:
    sampler: reverb_types.SelectorType = reverb.selectors.Uniform()
    remover: reverb_types.SelectorType = reverb.selectors.Fifo()
    max_size: int = 100000
    rate_limiter: rate_limiters.RateLimiter = None
    max_times_sampled: int = 0


class OffPolicyDataServer(DataServer):
    def __init__(
        self, config: OffPolicyDataServerConfig = OffPolicyDataServerConfig()
    ) -> None:
        """_summary_

        Args:
            config : _description_.
        """

        self.config = config

    def table(
        self,
        table_key: str,
        environment_spec: specs.MAEnvironmentSpec,
        extras_spec: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """_summary_

        Args:
            table_key : _description_
            environment_spec : _description_
            extras_spec : _description_
            builder : _description_
        Returns:
            _description_
        """
        table = reverb.Table(
            name=table_key,
            sampler=self.config.sampler,
            remover=self.config.remover,
            max_size=self.config.max_size,
            rate_limiter=builder.store.rate_limiter_fn(),
            signature=builder.store.adder_signature_fn(environment_spec, extras_spec),
            max_times_sampled=self.config.max_times_sampled,
        )
        return table

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return OffPolicyDataServerConfig

    @staticmethod
    def required_components() -> List[Type[Callback]]:
        """List of other Components required in the system for this Component to function.

        RateLimiter required to set up builder.store.rate_limiter_fn.

        Returns:
            List of required component classes.
        """
        return DataServer.required_components() + [RateLimiter]


@dataclass
class OnPolicyDataServerConfig:
    max_queue_size: int = 1000


class OnPolicyDataServer(DataServer):
    def __init__(
        self,
        config: OnPolicyDataServerConfig = OnPolicyDataServerConfig(),
    ) -> None:
        """_summary_

        Args:
            config : _description_.
        """

        self.config = config

    def table(
        self,
        table_key: str,
        environment_spec: specs.MAEnvironmentSpec,
        extras_spec: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """_summary_

        Args:
            table_key : _description_
            environment_spec : _description_
            extras_spec : _description_
            builder : _description_
        Returns:
            _description_
        """
        if hasattr(builder.store.global_config, "sequence_length"):
            signature = builder.store.adder_signature_fn(
                environment_spec,
                builder.store.global_config.sequence_length,
                extras_spec,
            )
        else:
            signature = builder.store.adder_signature_fn(environment_spec, extras_spec)
        table = reverb.Table.queue(
            name=table_key,
            max_size=self.config.max_queue_size,
            signature=signature,
        )
        return table

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return OnPolicyDataServerConfig
