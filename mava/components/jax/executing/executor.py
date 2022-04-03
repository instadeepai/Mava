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
from typing import Any, Dict, List, Union

import numpy as np
import jax

from mava.components.jax import Component
from mava.core_jax import SystemBuilder, SystemExecutor
from mava.utils import enums
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num


@dataclass
class ExecutorProcessConfig:
    network_sampling_setup: Union[
        List, enums.NetworkSampler
    ] = enums.NetworkSampler.fixed_agent_networks


class DefaultFeedforwardExecutor(Component):
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
        builder.config.agents = sort_str_num(
            builder.config.environment_spec.get_agent_ids()
        )

        if not isinstance(network_sampling_setup, list):
            if network_sampling_setup == enums.NetworkSampler.fixed_agent_networks:
                # if no network_sampling_setup is specified, assign a single network
                # to all agents of the same type if weights are shared
                # else assign seperate networks to each agent
                builder.config.agent_net_keys = {
                    agent: f"network_{agent.split('_')[0]}"
                    if self.config.shared_weights
                    else f"network_{agent}"
                    for agent in builder.config.agents
                }
                builder.config.network_sampling_setup = [
                    [
                        builder.config.agent_net_keys[key]
                        for key in sort_str_num(builder.config.agent_net_keys.keys())
                    ]
                ]
            elif network_sampling_setup == enums.NetworkSampler.random_agent_networks:
                """Create N network policies, where N is the number of agents. Randomly
                select policies from this sets for each agent at the start of a
                episode. This sampling is done with replacement so the same policy
                can be selected for more than one agent for a given episode."""
                if builder.config.shared_weights:
                    raise ValueError(
                        "Shared weights cannot be used with random policy per agent"
                    )
                builder.config.agent_net_keys = {
                    builder.config.agents[i]: f"network_{i}"
                    for i in range(len(builder.config.agents))
                }

                builder.config.network_sampling_setup = [
                    [
                        [builder.config.agent_net_keys[key]]
                        for key in sort_str_num(builder.config.agent_net_keys.keys())
                    ]
                ]
            else:
                raise ValueError(
                    "network_sampling_setup must be a dict or fixed_agent_networks"
                )
        else:
            # if a dictionary is provided, use network_sampling_setup to determine setup
            _, builder.config.agent_net_keys = sample_new_agent_keys(
                builder.config.agents,
                builder.config.network_sampling_setup,
            )

        # Check that the environment and agent_net_keys has the same amount of agents
        sample_length = len(builder.config.network_sampling_setup[0])
        agent_ids = builder.config.environment_spec.get_agent_ids()
        assert len(agent_ids) == len(builder.config.agent_net_keys.keys())

        # Check if the samples are of the same length and that they perfectly fit
        # into the total number of agents
        assert len(builder.config.agent_net_keys.keys()) % sample_length == 0
        for i in range(1, len(builder.config.network_sampling_setup)):
            assert len(builder.config.network_sampling_setup[i]) == sample_length

        # Get all the unique agent network keys
        all_samples = []
        for sample in builder.config.network_sampling_setup:
            all_samples.extend(sample)
        builder.config.unique_net_keys = list(sort_str_num(list(set(all_samples))))

        # Create mapping from ints to networks
        builder.config.net_keys_to_ids = {
            net_key: i for i, net_key in enumerate(builder.config.unique_net_keys)
        }

    # Start executor
    def on_execution_init_start(self, executor: SystemExecutor) -> None:
        executor.config.policy_networks = executor.config.network_factory()[
            "policy_networks"
        ]

    # Observe first
    def on_execution_observe_first(self, executor: SystemExecutor) -> None:
        if not executor.config.adder:
            return

        "Select new networks from the sampler at the start of each episode."
        agents = sort_str_num(list(executor.config.agent_net_keys.keys()))
        (
            executor.config.network_int_keys_extras,
            executor.config.agent_net_keys,
        ) = sample_new_agent_keys(
            agents,
            executor.config.network_sampling_setup,
            executor.config.net_keys_to_ids,
        )
        executor.config.extras[
            "network_int_keys"
        ] = executor.config.network_int_keys_extras
        executor.config.adder.add_first(
            executor.config.timestep, executor.config.extras
        )

    # Observe
    def on_execution_observe(self, executor: SystemExecutor) -> None:
        if not executor.config.adder:
            return

        actions_info = executor.config.actions_info
        policies_info = executor.config.policies_info

        adder_actions: Dict[str, Any] = {}
        for agent in actions_info.keys():
            adder_actions[agent] = {
                "actions_info": actions_info[agent],
                "policy_info": policies_info[agent],
            }

        executor.config.next_extras[
            "network_int_keys"
        ] = executor.config.network_int_keys_extras
        executor.config.adder.add(
            adder_actions, executor.config.next_timestep, executor.config.next_extras
        )

    # Update the executor variables.
    def on_execution_update(self, executor: SystemExecutor) -> None:
        """Update the policy variables."""
        if executor.config.executor_parameter_client:
            executor.config.executor_parameter_client.get_async()

    # Select actions
    def on_execution_select_actions(self, executor: SystemExecutor) -> None:
        """Summary"""
        executor.config.actions_info = {}
        executor.config.policies_info = {}
        for agent, observation in executor.config.observations.items():
            action_info, policy_info = executor.select_action(agent, observation)
            executor.config.actions_info[agent] = action_info
            executor.config.policies_info[agent] = policy_info

    # Select action
    def on_execution_select_action_compute(self, executor: SystemExecutor) -> None:
        """Summary"""
        agent = executor.config.agent
        policy = executor.config.policy_networks[
            executor.config.agent_net_keys[agent]
        ]

        observation = executor.config.observation.observation.reshape((1, -1))
        rng_key, executor.config.key = jax.random.split(executor.config.key)
        executor.config.action_info, executor.config.policy_info = policy.get_action(observation, rng_key)

    @property
    def name(self) -> str:
        """_summary_"""
        return "executor"
