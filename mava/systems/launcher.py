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

"""Commonly used dataset components for system builders"""
from typing import Dict, List, Optional, Union, Any

import launchpad as lp

from mava.core import SystemBuilder
from mava.utils import enums
from mava.utils.sort_utils import sort_str_num, sample_new_agent_keys
from mava.utils import lp_utils


class NodeType:
    reverb = lp.ReverbNode
    corrier = lp.CourierNode


class Launcher:
    def __init__(
        self,
        builder: SystemBuilder,
        name: str = "System",
        single_process: bool = False,
        num_executors: int = 1,
        network_sampling_setup: Union[
            List, enums.NetworkSampler
        ] = enums.NetworkSampler.fixed_agent_networks,
        trainer_networks: Union[
            Dict[str, List], enums.Trainer
        ] = enums.Trainer.single_trainer,
        nodes_on_gpu: List = [],
        termination_condition: Optional[Dict[str, int]] = None,
    ) -> None:
        """summary"""

        self.name = name
        self.single_process = single_process

        if not single_process:
            self.program = lp.Program(name=name)
            self.nodes_on_gpu = nodes_on_gpu

        self.num_executors = num_executors
        self.termination_condition = termination_condition

        builder.network_sampling_setup = network_sampling_setup
        builder.trainer_networks = trainer_networks

        # Setup agent networks and network sampling setup
        agents = sort_str_num(builder.environment_spec.get_agent_ids())

        if type(network_sampling_setup) is not list:
            if network_sampling_setup == enums.NetworkSampler.fixed_agent_networks:
                # if no network_sampling_setup is fixed, use shared_weights to
                # determine setup
                builder.agent_net_keys = {
                    agent: "network_0" if builder.shared_weights else f"network_{i}"
                    for i, agent in enumerate(agents)
                }
                builder.network_sampling_setup = [
                    [
                        builder.agent_net_keys[key]
                        for key in sort_str_num(builder.agent_net_keys.keys())
                    ]
                ]
            elif network_sampling_setup == enums.NetworkSampler.random_agent_networks:
                """Create N network policies, where N is the number of agents. Randomly
                select policies from this set for each agent at the start of a
                episode. This sampling is done with replacement so the same policy
                can be selected for more than one agent for a given episode."""
                if builder.shared_weights:
                    raise ValueError(
                        "Shared weights cannot be used with random policy per agent"
                    )
                builder.agent_net_keys = {
                    agents[i]: f"network_{i}" for i in range(len(agents))
                }
                builder.network_sampling_setup = [
                    [
                        [builder.agent_net_keys[key]]
                        for key in sort_str_num(builder.agent_net_keys.keys())
                    ]
                ]
            else:
                raise ValueError(
                    "network_sampling_setup must be a dict or fixed_agent_networks"
                )

        else:
            # if a dictionary is provided, use network_sampling_setup to determine setup
            _, builder.agent_net_keys = sample_new_agent_keys(
                agents,
                builder.network_sampling_setup,  # type: ignore
            )

        # Check that the environment and agent_net_keys has the same amount of agents
        sample_length = len(builder.network_sampling_setup[0])  # type: ignore
        assert len(builder.environment_spec.get_agent_ids()) == len(
            builder.agent_net_keys.keys()
        )

        # Check if the samples are of the same length and that they perfectly fit
        # into the total number of agents
        assert len(builder.agent_net_keys.keys()) % sample_length == 0
        for i in range(1, len(self.network_sampling_setup)):  # type: ignore
            assert len(builder.network_sampling_setup[i]) == sample_length  # type: ignore

        # Get all the unique agent network keys
        all_samples = []
        for sample in self.network_sampling_setup:  # type: ignore
            all_samples.extend(sample)
        unique_net_keys = list(sort_str_num(list(set(all_samples))))

        # Create mapping from ints to networks
        builder.net_keys_to_ids = {
            net_key: i for i, net_key in enumerate(unique_net_keys)
        }

        # Setup trainer_networks
        if type(trainer_networks) is not dict:
            if trainer_networks == enums.Trainer.single_trainer:
                builder.trainer_networks = {"trainer": unique_net_keys}
            elif trainer_networks == enums.Trainer.one_trainer_per_network:
                builder.trainer_networks = {
                    f"trainer_{i}": [unique_net_keys[i]]
                    for i in range(len(unique_net_keys))
                }
            else:
                raise ValueError(
                    "trainer_networks does not support this enums setting."
                )

        # Get all the unique trainer network keys
        all_trainer_net_keys = []
        for trainer_nets in self.trainer_networks.values():
            all_trainer_net_keys.extend(trainer_nets)
        unique_trainer_net_keys = sort_str_num(list(set(all_trainer_net_keys)))

        # Check that all agent_net_keys are in trainer_networks
        assert unique_net_keys == unique_trainer_net_keys
        # Setup specs for each network
        self._net_spec_keys = {}
        for i in range(len(unique_net_keys)):
            self._net_spec_keys[unique_net_keys[i]] = agents[i % len(agents)]

        # Setup table_network_config
        builder.table_network_config = {}
        for trainer_key in builder.trainer_networks.keys():
            most_matches = 0
            trainer_nets = builder.trainer_networks[trainer_key]
            for sample in builder.network_sampling_setup:  # type: ignore
                matches = 0
                for entry in sample:
                    if entry in trainer_nets:
                        matches += 1
                if most_matches < matches:
                    matches = most_matches
                    builder.table_network_config[trainer_key] = sample

        # set launch builder
        self.build = builder

        # TODO (Arnu): figure out how to handle extra spec.

        # extra_specs = {}
        # # if issubclass(executor_fn, executors.RecurrentExecutor):
        # #     extra_specs = self._get_extra_specs()

        # int_spec = specs.DiscreteArray(len(unique_net_keys))
        # agents = builder.environment_spec.get_agent_ids()
        # net_spec = {"network_keys": {agent: int_spec for agent in agents}}
        # extra_specs.update(net_spec)

    def add(
        self,
        node_fn: Any,
        arguments: Any = [],
        node_type: Union[lp.ReverbNode, lp.CourierNode] = NodeType.corrier,
        name: str = "Node",
    ) -> Any:
        # Create a list of arguments
        if type(arguments) is not list:
            arguments = [arguments]

        if self._single_process:
            raise NotImplementedError("Single process launching not implemented yet.")
        else:
            with self._program.group(name):
                node = self._program.add_node(node_type(node_fn, *arguments))
            return node

    def launch(self) -> None:
        if self._single_process:
            raise NotImplementedError("Single process launching not implemented yet.")
        else:
            local_resources = lp_utils.to_device(
                program_nodes=self._program.groups.keys(),
                nodes_on_gpu=self._nodes_on_gpu,
            )

            lp.launch(
                self._program,
                lp.LaunchType.LOCAL_MULTI_PROCESSING,
                terminal="current_terminal",
                local_resources=local_resources,
            )