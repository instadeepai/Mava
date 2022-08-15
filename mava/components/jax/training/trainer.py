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

import abc
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

from mava.components.jax import Component
from mava.core_jax import SystemBuilder, SystemTrainer
from mava.utils.sort_utils import sort_str_num


class BaseTrainerInit(Component):
    @abc.abstractmethod
    def __init__(
        self,
        config: Any,
    ):
        """Component sets up trainer networks.

        Args:
            config : a dataclass specifying the component parameters.
        """
        self.config = config

    @abc.abstractmethod
    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Set up the networks during the build."""
        pass

    @abc.abstractmethod
    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Set up trainer agents."""
        pass

    @staticmethod
    def name() -> str:
        """Component name."""

        return "trainer_init"


class SingleTrainerInit(BaseTrainerInit):
    def __init__(self, config: SimpleNamespace = SimpleNamespace()):
        """Initialises a single trainer.

        Single trainer is used to train all networks.

        Args:
            config : a dataclass specifying the component parameters.
        """
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Assigns trainers to networks for training.

        Args:
            builder : the system builder
        Raises:
            ValueError: Raises an error when trainer_networks is not
                        set to single_trainer.
        """
        unique_net_keys = builder.store.unique_net_keys

        # Setup trainer_networks

        builder.store.trainer_networks = {"trainer": unique_net_keys}

        # Get all the unique trainer network keys
        all_trainer_net_keys = []
        for trainer_nets in builder.store.trainer_networks.values():
            all_trainer_net_keys.extend(trainer_nets)
        unique_trainer_net_keys = sort_str_num(list(set(all_trainer_net_keys)))

        # Check that all agent_net_keys are in trainer_networks
        assert unique_net_keys == unique_trainer_net_keys
        # Setup specs for each network
        builder.store.net_spec_keys = {}
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

        builder.store.networks = builder.store.network_factory()

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Set up trainer agents.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """
        # Convert network keys for the trainer.
        trainer.store.trainer_table_entry = trainer.store.table_network_config[
            trainer.store.trainer_id
        ]
        trainer.store.trainer_agents = trainer.store.agents[
            : len(trainer.store.trainer_table_entry)
        ]
        trainer.store.trainer_agent_net_keys = {
            agent: trainer.store.trainer_table_entry[a_i]
            for a_i, agent in enumerate(trainer.store.trainer_agents)
        }


class OneTrainerPerNetworkInit(BaseTrainerInit):
    def __init__(self, config: SimpleNamespace = SimpleNamespace()):
        """Initialises a multiple trainers.

        Different trainer will be dedicated to training each network.

        Args:
            config : a dataclass specifying the component parameters.
        """
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Set up trainer networks.

        Args:
            builder : the system builder
        Raises:
            ValueError: Raises an error when trainer_networks is not
                        set to one_trainer_per_network.
        """
        unique_net_keys = builder.store.unique_net_keys

        # Setup trainer_networks
        builder.store.trainer_networks = {
            f"trainer_{i}": [unique_net_keys[i]] for i in range(len(unique_net_keys))
        }

        # Get all the unique trainer network keys
        all_trainer_net_keys = []
        for trainer_nets in builder.store.trainer_networks.values():
            all_trainer_net_keys.extend(trainer_nets)
        unique_trainer_net_keys = sort_str_num(list(set(all_trainer_net_keys)))

        # Check that all agent_net_keys are in trainer_networks
        assert unique_net_keys == unique_trainer_net_keys
        # Setup specs for each network
        builder.store.net_spec_keys = {}
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

        builder.store.networks = builder.store.network_factory()

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Set up trainer agents.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """
        # Convert network keys for the trainer.
        trainer.store.trainer_table_entry = trainer.store.table_network_config[
            trainer.store.trainer_id
        ]
        trainer.store.trainer_agents = trainer.store.agents[
            : len(trainer.store.trainer_table_entry)
        ]
        trainer.store.trainer_agent_net_keys = {
            agent: trainer.store.trainer_table_entry[a_i]
            for a_i, agent in enumerate(trainer.store.trainer_agents)
        }


@dataclass
class CustomTrainerInitConfig:
    trainer_networks: Dict = field(default_factory=lambda: {})


class CustomTrainerInit(BaseTrainerInit):
    def __init__(self, config: CustomTrainerInitConfig = CustomTrainerInitConfig()):
        """Initialises custom trainers.

        Custom trainer network configuration can be given as a dictionary
        assigning specific trainers to specific networks.

        Args:
            config : a dataclass specifying the component parameters.
        """

        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Assigns trainers to networks for training.

        Args:
            builder : the system builder
        Raises:
            ValueError: Raises an error when trainer_networks is not
                        passed in as a dictionary.
        """
        trainer_networks = self.config.trainer_networks
        unique_net_keys = builder.store.unique_net_keys

        # Setup trainer_networks
        if not isinstance(trainer_networks, dict) or trainer_networks == {}:

            raise ValueError("trainer_networks must be a non-empty dictionary.")

        builder.store.trainer_networks = trainer_networks

        # Get all the unique trainer network keys
        all_trainer_net_keys = []
        for trainer_nets in builder.store.trainer_networks.values():
            all_trainer_net_keys.extend(trainer_nets)
        unique_trainer_net_keys = sort_str_num(list(set(all_trainer_net_keys)))

        # Check that all agent_net_keys are in trainer_networks
        assert unique_net_keys == unique_trainer_net_keys
        # Setup specs for each network
        builder.store.net_spec_keys = {}
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

        builder.store.networks = builder.store.network_factory()

    def on_training_utility_fns(self, trainer: SystemTrainer) -> None:
        """Set up and store trainer agents.

        Args:
            trainer: SystemTrainer.

        Returns:
            None.
        """
        # Convert network keys for the trainer.
        trainer.store.trainer_table_entry = trainer.store.table_network_config[
            trainer.store.trainer_id
        ]
        trainer.store.trainer_agents = trainer.store.agents[
            : len(trainer.store.trainer_table_entry)
        ]
        trainer.store.trainer_agent_net_keys = {
            agent: trainer.store.trainer_table_entry[a_i]
            for a_i, agent in enumerate(trainer.store.trainer_agents)
        }

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return CustomTrainerInitConfig
