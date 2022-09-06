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
from typing import Any, Callable, Dict, List, Optional

import reverb

from mava import specs
from mava.components.jax import Component
from mava.core_jax import SystemBuilder

# Check network type
from mava.systems.jax.madqn.DQNNetworks import DQNNetworks
from mava.utils import enums
from mava.utils.builder_utils import convert_specs
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

            # DQN requires next extras for transition adder and IPPO does not
            if isinstance(
                builder.store.networks["networks"]["network_agent"], DQNNetworks
            ):

                next_extras_specs = convert_specs(
                    builder.store.agent_net_keys,
                    builder.store.next_extras_specs,
                    num_networks,
                )

                table = self.table(
                    table_key=table_key,
                    environment_specs=env_specs,
                    extras_specs=extras_specs,
                    next_extras_specs=next_extras_specs,
                    builder=builder,
                )
            else:
                table = self.table(  # type: ignore
                    table_key, env_specs, extras_specs, builder  # type: ignore
                )

            data_tables.append(table)
        return data_tables

    @abc.abstractmethod
    def table(
        self,
        table_key: str,
        environment_specs: specs.MAEnvironmentSpec,
        extras_specs: Dict[str, Any],
        next_extras_specs: Dict[str, Any],
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
    max_size: int = 100000
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
        environment_specs: specs.MAEnvironmentSpec,
        extras_specs: Dict[str, Any],
        next_extras_specs: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """_summary_

        Args:
            table_key : _description_
            environment_specs : _description_
            extras_specs : _description_
            next_extras_specs: _description_
            builder : _description_
        Returns:
            _description_
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
            signature=builder.store.adder_signature_fn(
                environment_specs, extras_specs, next_extras_specs
            ),
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

    def table(  # type: ignore
        self,
        table_key: str,
        environment_specs: specs.MAEnvironmentSpec,
        extras_specs: Dict[str, Any],
        builder: SystemBuilder,
    ) -> reverb.Table:
        """_summary_

        Args:
            table_key : _description_
            environment_specs : _description_
            extras_specs : _description_
            builder : _description_
        Returns:
            _description_
        """
        if builder.store.__dict__.get("sequence_length"):
            signature = builder.store.adder_signature_fn(
                environment_specs, builder.store.sequence_length, extras_specs
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

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return OnPolicyDataServerConfig
