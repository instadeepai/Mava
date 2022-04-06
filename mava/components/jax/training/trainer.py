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

"""Trainer components for system builders."""

from dataclasses import dataclass
from typing import Any, Dict, List, Union

from mava.components.jax import Component
from mava.core_jax import SystemBuilder
from mava.utils import enums
from mava.utils.sort_utils import sort_str_num


@dataclass
class TrainerProcessConfig:
    trainer_networks: Union[
        Dict[str, List], enums.Trainer
    ] = enums.Trainer.single_trainer


class TrainerProcess(Component):
    def __init__(self, config: TrainerProcessConfig = TrainerProcessConfig()):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        Raises:
            ValueError: _description_
        """
        trainer_networks = self.config.trainer_networks
        unique_net_keys = builder.store.unique_net_keys

        # Setup trainer_networks
        if not isinstance(trainer_networks, dict):
            if trainer_networks == enums.Trainer.single_trainer:
                builder.store.trainer_networks = {"trainer": unique_net_keys}
            elif trainer_networks == enums.Trainer.one_trainer_per_network:
                builder.store.trainer_networks = {
                    f"trainer_{i}": [unique_net_keys[i]]
                    for i in range(len(unique_net_keys))
                }
            else:
                raise ValueError(
                    "trainer_networks does not support this enums setting."
                )
        else:
            builder.store.trainer_networks = trainer_networks

        # Get all the unique trainer network keys
        all_trainer_net_keys = []
        for trainer_nets in builder.store.trainer_networks.values():
            all_trainer_net_keys.extend(trainer_nets)
        unique_trainer_net_keys = sort_str_num(list(set(all_trainer_net_keys)))

        # Check that all agent_net_keys are in trainer_networks
        assert unique_net_keys == unique_trainer_net_keys
        # Setup specs for each network
        self._net_spec_keys: Dict[str, Any] = {}
        for i in range(len(unique_net_keys)):
            builder.store.net_spec_keys[unique_net_keys[i]] = builder.store.agents[
                i % len(builder.store.agents)
            ]

        # Setup table_network_config
        builder.store.table_network_config = {}
        for trainer_key in builder.store.trainer_networks.keys():
            most_matches = 0
            trainer_nets = builder.store.trainer_networks[trainer_key]
            for sample in builder.store.network_sampling_setup:
                matches = 0
                for entry in sample:
                    if entry in trainer_nets:
                        matches += 1
                if most_matches < matches:
                    matches = most_matches
                    builder.store.table_network_config[trainer_key] = sample

    def on_building_trainer_start(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.store.networks = builder.store.network_factory(
            environment_spec=builder.store.environment_spec,
            agent_net_keys=builder.store.agent_net_keys,
            net_spec_keys=builder.store.net_spec_keys,
        )

    def name(self) -> str:
        """_summary_"""
        return "trainer"
