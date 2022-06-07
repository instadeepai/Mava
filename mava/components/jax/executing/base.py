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
from typing import Callable, Optional

from mava.components.jax import Component
from mava.core_jax import SystemBuilder, SystemExecutor
from mava.utils import enums
from mava.utils.id_utils import EntityId
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num


@dataclass
class ExecutorInitConfig:
    interval: Optional[dict] = None


class ExecutorInit(Component):
    def __init__(self, config: ExecutorInitConfig = ExecutorInitConfig()):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_init(self, builder: SystemBuilder) -> None:
        """Summary"""
        # Setup agent networks and network sampling setup
        network_sampling_setup = self.config.network_sampling_setup
        builder.store.agents = sort_str_num(
            builder.store.environment_spec.get_agent_ids()
        )

        if not isinstance(network_sampling_setup, list):
            if network_sampling_setup == enums.NetworkSampler.fixed_agent_networks:
                # if no network_sampling_setup is specified, assign a single network
                # to all agents of the same type if weights are shared
                # else assign seperate networks to each agent
                builder.store.agent_net_keys = {
                    agent: f"network_{EntityId.from_string(agent).type}"
                    if self.config.shared_weights
                    else f"network_{agent}"
                    for agent in builder.store.agents
                }
                builder.store.network_sampling_setup = [
                    [
                        builder.store.agent_net_keys[key]
                        for key in sort_str_num(builder.store.agent_net_keys.keys())
                    ]
                ]
            elif network_sampling_setup == enums.NetworkSampler.random_agent_networks:
                """Create N network policies, where N is the number of agents. Randomly
                select policies from this sets for each agent at the start of a
                episode. This sampling is done with replacement so the same policy
                can be selected for more than one agent for a given episode."""
                if builder.store.shared_weights:
                    raise ValueError(
                        "Shared weights cannot be used with random policy per agent"
                    )
                builder.store.agent_net_keys = {
                    builder.store.agents[i]: f"network_{i}"
                    for i in range(len(builder.store.agents))
                }

                builder.store.network_sampling_setup = [
                    [
                        [builder.store.agent_net_keys[key]]
                        for key in sort_str_num(builder.store.agent_net_keys.keys())
                    ]
                ]
            else:
                raise ValueError(
                    "network_sampling_setup must be a dict or fixed_agent_networks"
                )
        else:
            # if a dictionary is provided, use network_sampling_setup to determine setup
            _, builder.store.agent_net_keys = sample_new_agent_keys(
                builder.store.agents,
                builder.store.network_sampling_setup,
            )

        # Check that the environment and agent_net_keys has the same amount of agents
        sample_length = len(builder.store.network_sampling_setup[0])
        agent_ids = builder.store.environment_spec.get_agent_ids()
        assert len(agent_ids) == len(builder.store.agent_net_keys.keys())

        # Check if the samples are of the same length and that they perfectly fit
        # into the total number of agents
        assert len(builder.store.agent_net_keys.keys()) % sample_length == 0
        for i in range(1, len(builder.store.network_sampling_setup)):
            assert len(builder.store.network_sampling_setup[i]) == sample_length

        # Get all the unique agent network keys
        all_samples = []
        for sample in builder.store.network_sampling_setup:
            all_samples.extend(sample)
        builder.store.unique_net_keys = list(sort_str_num(list(set(all_samples))))

        # Create mapping from ints to networks
        builder.store.net_keys_to_ids = {
            net_key: i for i, net_key in enumerate(builder.store.unique_net_keys)
        }

        builder.store.networks = builder.store.network_factory()

    def on_execution_init_start(self, executor: SystemExecutor) -> None:
        """_summary_"""
        executor._interval = self.config.interval  # type: ignore

    @staticmethod
    def name() -> str:
        """_summary_"""
        return "executor_init"

    @staticmethod
    def config_class() -> Optional[Callable]:
        """Config class used for component.

        Returns:
            config class/dataclass for component.
        """
        return ExecutorInitConfig
