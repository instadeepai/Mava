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

"""Base components for system builder"""
import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Union

import jax
import jax.numpy as jnp

from mava.components.jax import Component
from mava.core_jax import SystemBuilder
from mava.utils import enums
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num


@dataclass
class ExecutorProcessConfig:
    network_sampling_setup: Union[
        List, enums.NetworkSampler
    ] = enums.NetworkSampler.fixed_agent_networks
    shared_weights: bool = True


class BaseExecutorProcess(Component):
    def __init__(self, config: ExecutorProcessConfig = ExecutorProcessConfig()):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_init(self, builder: SystemBuilder) -> None:
        """Summary"""

        # Setup agent networks and network sampling setup
        network_sampling_setup = self.config.network_sampling_setup
        builder.attr.agents = sort_str_num(
            builder.attr.environment_spec.get_agent_ids()
        )
        builder.attr.shared_weights = self.config.shared_weights

        if not isinstance(network_sampling_setup, list):
            if network_sampling_setup == enums.NetworkSampler.fixed_agent_networks:
                # if no network_sampling_setup is specified, assign a single network
                # to all agents of the same type if weights are shared
                # else assign seperate networks to each agent
                builder.attr.agent_net_keys = {
                    agent: f"network_{agent.split('_')[0]}"
                    if self.config.shared_weights
                    else f"network_{agent}"
                    for agent in builder.attr.agents
                }
                builder.attr.network_sampling_setup = [
                    [
                        builder.attr.agent_net_keys[key]
                        for key in sort_str_num(builder.attr.agent_net_keys.keys())
                    ]
                ]
            elif network_sampling_setup == enums.NetworkSampler.random_agent_networks:
                """Create N network policies, where N is the number of agents. Randomly
                select policies from this sets for each agent at the start of a
                episode. This sampling is done with replacement so the same policy
                can be selected for more than one agent for a given episode."""
                if builder.attr.shared_weights:
                    raise ValueError(
                        "Shared weights cannot be used with random policy per agent"
                    )
                builder.attr.agent_net_keys = {
                    builder.attr.agents[i]: f"network_{i}"
                    for i in range(len(builder.attr.agents))
                }

                builder.attr.network_sampling_setup = [
                    [
                        [builder.attr.agent_net_keys[key]]
                        for key in sort_str_num(builder.attr.agent_net_keys.keys())
                    ]
                ]
            else:
                raise ValueError(
                    "network_sampling_setup must be a dict or fixed_agent_networks"
                )
        else:
            # if a dictionary is provided, use network_sampling_setup to determine setup
            _, builder.attr.agent_net_keys = sample_new_agent_keys(
                builder.attr.agents,
                builder.attr.network_sampling_setup,
            )

        # Check that the environment and agent_net_keys has the same amount of agents
        sample_length = len(builder.attr.network_sampling_setup[0])
        agent_ids = builder.attr.environment_spec.get_agent_ids()
        assert len(agent_ids) == len(builder.attr.agent_net_keys.keys())

        # Check if the samples are of the same length and that they perfectly fit
        # into the total number of agents
        assert len(builder.attr.agent_net_keys.keys()) % sample_length == 0
        for i in range(1, len(builder.attr.network_sampling_setup)):
            assert len(builder.attr.network_sampling_setup[i]) == sample_length

        # Get all the unique agent network keys
        all_samples = []
        for sample in builder.attr.network_sampling_setup:
            all_samples.extend(sample)
        builder.attr.unique_net_keys = list(sort_str_num(list(set(all_samples))))

        # Create mapping from ints to networks
        builder.attr.net_keys_to_ids = {
            net_key: i for i, net_key in enumerate(builder.attr.unique_net_keys)
        }

    def on_building_executor_start(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.networks = builder.attr.network_factory(
            environment_spec=builder.attr.environment_spec,
            agent_net_keys=builder.attr.agent_net_keys,
            net_spec_keys=builder.attr.net_spec_keys,
        )

    @abc.abstractmethod
    def on_building_executor(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """

    def name(self) -> str:
        """_summary_"""
        return "executor"


@dataclass
class TrainerProcessConfig:
    trainer_networks: Union[
        Dict[str, List], enums.Trainer
    ] = enums.Trainer.single_trainer


class BaseTrainerProcess(Component):
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
        unique_net_keys = builder.attr.unique_net_keys

        # Setup trainer_networks
        if not isinstance(trainer_networks, dict):
            if trainer_networks == enums.Trainer.single_trainer:
                builder.attr.trainer_networks = {"trainer": unique_net_keys}
            elif trainer_networks == enums.Trainer.one_trainer_per_network:
                builder.attr.trainer_networks = {
                    f"trainer_{i}": [unique_net_keys[i]]
                    for i in range(len(unique_net_keys))
                }
            else:
                raise ValueError(
                    "trainer_networks does not support this enums setting."
                )
        else:
            builder.attr.trainer_networks = trainer_networks

        # Get all the unique trainer network keys
        all_trainer_net_keys = []
        for trainer_nets in builder.attr.trainer_networks.values():
            all_trainer_net_keys.extend(trainer_nets)
        unique_trainer_net_keys = sort_str_num(list(set(all_trainer_net_keys)))

        # Check that all agent_net_keys are in trainer_networks
        assert unique_net_keys == unique_trainer_net_keys
        # Setup specs for each network
        self._net_spec_keys: Dict[str, Any] = {}
        for i in range(len(unique_net_keys)):
            builder.attr.net_spec_keys[unique_net_keys[i]] = builder.attr.agents[
                i % len(builder.attr.agents)
            ]

        # Setup table_network_config
        builder.attr.table_network_config = {}
        for trainer_key in builder.attr.trainer_networks.keys():
            most_matches = 0
            trainer_nets = builder.attr.trainer_networks[trainer_key]
            for sample in builder.attr.network_sampling_setup:
                matches = 0
                for entry in sample:
                    if entry in trainer_nets:
                        matches += 1
                if most_matches < matches:
                    matches = most_matches
                    builder.attr.table_network_config[trainer_key] = sample

    def on_building_trainer_start(self, builder: SystemBuilder) -> None:
        """_summary_"""
        builder.attr.networks = builder.attr.network_factory(
            environment_spec=builder.attr.environment_spec,
            agent_net_keys=builder.attr.agent_net_keys,
            net_spec_keys=builder.attr.net_spec_keys,
        )

    @abc.abstractmethod
    def on_building_trainer(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """

    def name(self) -> str:
        """_summary_"""
        return "trainer"


@dataclass
class ParameterServerProcessConfig:
    random_param: int = 5


class BaseParameterServerProcess(Component):
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

    @abc.abstractmethod
    def on_building_parameter_server(self, builder: SystemBuilder) -> None:
        """_summary_

        Args:
            builder : _description_
        """

    def name(self) -> str:
        """_summary_"""
        return "parameter_server"
