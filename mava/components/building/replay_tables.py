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
import copy

import reverb
from reverb import reverb_types

from mava import specs
from mava.callbacks import Callback
from mava.systems.building import BaseSystemBuilder
from mava.adders import reverb as reverb_adders
from mava.utils.sort_utils import sort_str_num


class ReplayTables(Callback):
    def __init__(
        self,
        name: str,
        agent_net_keys: Dict[str, str],
        table_network_config: Dict[str, List],
        sampler: reverb_types.SelectorType,
        remover: reverb_types.SelectorType,
        max_size: int,
        max_times_sampled: int = 0,
        extensions: Sequence[TableExtensionBase] = (),
    ):
        """[summary]

        Args:
            name (str): [description]
            agent_net_keys (Dict[str, str]): [description]
            table_network_config (Dict[str, List]): [description]
            sampler (reverb_types.SelectorType): [description]
            remover (reverb_types.SelectorType): [description]
            max_size (int): [description]
            max_times_sampled (int, optional): [description]. Defaults to 0.
            extensions (Sequence[TableExtensionBase], optional): [description]. Defaults to ().
        """

        self.name = name
        self.agent_net_keys = agent_net_keys
        self.table_network_config = table_network_config
        self.sampler = sampler
        self.remover = remover
        self.max_size = max_size
        self.max_times_sampled = max_times_sampled
        self.extensions = extensions

    def on_building_make_tables(self, builder: BaseSystemBuilder):
        """[summary]

        Args:
            builder (BaseSystemBuilder): [description]
        """
        pass


class OffPolicyReplayTables(ReplayTables):
    def _covert_specs(self, spec: Dict[str, Any], num_networks: int) -> Dict[str, Any]:
        """[summary]

        Args:
            spec (Dict[str, Any]): [description]
            num_networks (int): [description]

        Returns:
            Dict[str, Any]: [description]
        """
        if type(spec) is not dict:
            return spec

        agents = sort_str_num(self.agent_net_keys.keys())[:num_networks]
        converted_spec: Dict[str, Any] = {}
        if agents[0] in spec.keys():
            for agent in agents:
                converted_spec[agent] = spec[agent]
        else:
            # For the extras
            for key in spec.keys():
                converted_spec[key] = self._covert_specs(spec[key], num_networks)
        return converted_spec

    def on_building_make_tables(self, builder: BaseSystemBuilder):
        # Create table per trainer
        replay_tables = []
        for trainer_id in range(len(self.table_network_config.keys())):
            # TODO (dries): Clean the below coverter code up.
            # Convert a Mava spec
            num_networks = len(self.table_network_config[f"trainer_{trainer_id}"])
            env_spec = copy.deepcopy(builder._env_spec)
            env_spec._specs = self._covert_specs(env_spec._specs, num_networks)

            env_spec._keys = list(sort_str_num(env_spec._specs.keys()))
            if env_spec.extra_specs is not None:
                env_spec.extra_specs = self._covert_specs(
                    env_spec.extra_specs, num_networks
                )
            extra_specs = self._covert_specs(
                builder._extra_specs,
                num_networks,
            )

            replay_tables.append(
                reverb.Table(
                    name=f"{self.replay_table_name}_{trainer_id}",
                    sampler=self.sampler,
                    remover=self.remover,
                    max_size=self.max_size,
                    rate_limiter=builder.rate_limiter_fn(),
                    signature=builder.adder_signature_fn(env_spec, extra_specs),
                )
            )

        builder.replay_tables = replay_tables