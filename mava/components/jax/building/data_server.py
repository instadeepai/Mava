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
from typing import Any, Callable, Dict, List

import reverb
from reverb import reverb_types

from mava import specs
from mava.components.jax import Component
from mava.core_jax import SystemBuilder
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

            next_extras_spec = covert_specs(
                builder.store.agent_net_keys,
                builder.store.next_extras_spec,
                num_networks,
            )

            table = self.table(
                table_key=table_key,
                environment_spec=env_spec,
                extras_spec=extras_spec,
                next_extras_spec=next_extras_spec,
                builder=builder,
            )
            data_tables.append(table)
        return data_tables

    @abc.abstractmethod
    def table(
        self,
        table_key: str,
        environment_spec: specs.MAEnvironmentSpec,
        extras_spec: Dict[str, Any],
        next_extras_spec: Dict[str, Any],
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


@dataclass
class OffPolicyDataServerConfig:
    sampler: reverb_types.SelectorType = reverb.item_selectors.Uniform
    remover: reverb_types.SelectorType = reverb.item_selectors.Fifo
    max_size: int = 1_000_000


class OffPolicyDataServer(DataServer):
    def __init__(
        self,
        config: OffPolicyDataServerConfig = OffPolicyDataServerConfig(),
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
        next_extras_spec: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """_summary_

        Args:
            table_key : _description_
            environment_spec : _description_
            extras_spec : _description_
            next_extras_spec: _description_
            builder : _description_
        Returns:
            _description_
        """
        table = reverb.Table(
            name=table_key,
            sampler=self.config.sampler(),
            remover=self.config.remover(),
            max_size=self.config.max_size,
            rate_limiter=builder.store.rate_limiter_fn(),
            signature=builder.store.adder_signature_fn(
                environment_spec, extras_spec, next_extras_spec
            ),
        )
        return table

    @staticmethod
    def config_class() -> Callable:
        """Return the configuration class for this component."""
        return OffPolicyDataServerConfig


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
        table = reverb.Table.queue(
            name=table_key,
            max_size=self.config.max_queue_size,
            signature=builder.store.adder_signature_fn(
                environment_spec, builder.store.sequence_length, extras_spec
            ),
        )
        return table

    @staticmethod
    def config_class() -> Callable:
        """Return the configuration class for this component."""
        return OnPolicyDataServerConfig
