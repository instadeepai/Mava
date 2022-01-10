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
from typing import Dict, List, Optional, Union

import launchpad as lp

from mava.callbacks import Callback
from mava.core import SystemBuilder
from mava.utils import enums
from mava.utils.sort_utils import sort_str_num, sample_new_agent_keys


class Distributor(Callback):
    def __init__(
        self,
        num_executors: int = 1,
        network_sampling_setup: Union[
            List, enums.NetworkSampler
        ] = enums.NetworkSampler.fixed_agent_networks,
        termination_condition: Optional[Dict[str, int]] = None,
    ) -> None:
        """summary"""

        self.num_executors = num_executors
        self.network_sampling_setup = network_sampling_setup
        self.termination_condition = termination_condition

    def on_building_init(self, builder: SystemBuilder) -> None:
        # Setup agent networks and network sampling setup
        agents = sort_str_num(builder.environment_spec.get_agent_ids())

        if type(self.network_sampling_setup) is not list:
            if self.network_sampling_setup == enums.NetworkSampler.fixed_agent_networks:
                # if no network_sampling_setup is fixed, use shared_weights to
                # determine setup
                builder.agent_net_keys = {
                    agent: "network_0" if builder.shared_weights else f"network_{i}"
                    for i, agent in enumerate(agents)
                }
                self.network_sampling_setup = [
                    [
                        builder.agent_net_keys[key]
                        for key in sort_str_num(builder.agent_net_keys.keys())
                    ]
                ]
            elif (
                self.network_sampling_setup
                == enums.NetworkSampler.random_agent_networks
            ):
                """Create N network policies, where N is the number of agents. Randomly
                select policies from this sets for each agent at the start of a
                episode. This sampling is done with replacement so the same policy
                can be selected for more than one agent for a given episode."""
                if builder.shared_weights:
                    raise ValueError(
                        "Shared weights cannot be used with random policy per agent"
                    )
                builder.agent_net_keys = {
                    agents[i]: f"network_{i}" for i in range(len(agents))
                }
                self.network_sampling_setup = [
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
                self.network_sampling_setup,  # type: ignore
            )

        # Check that the environment and agent_net_keys has the same amount of agents
        sample_length = len(self.network_sampling_setup[0])  # type: ignore
        assert len(builder.environment_spec.get_agent_ids()) == len(
            builder.agent_net_keys.keys()
        )

        # Check if the samples are of the same length and that they perfectly fit
        # into the total number of agents
        assert len(builder.agent_net_keys.keys()) % sample_length == 0
        for i in range(1, len(self.network_sampling_setup)):  # type: ignore
            assert len(self.network_sampling_setup[i]) == sample_length  # type: ignore

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
                self._trainer_networks = {"trainer": unique_net_keys}
            elif trainer_networks == enums.Trainer.one_trainer_per_network:
                self._trainer_networks = {
                    f"trainer_{i}": [unique_net_keys[i]]
                    for i in range(len(unique_net_keys))
                }
            else:
                raise ValueError(
                    "trainer_networks does not support this enums setting."
                )
        else:
            self._trainer_networks = trainer_networks  # type: ignore

        # Get all the unique trainer network keys
        all_trainer_net_keys = []
        for trainer_nets in self._trainer_networks.values():
            all_trainer_net_keys.extend(trainer_nets)
        unique_trainer_net_keys = sort_str_num(list(set(all_trainer_net_keys)))

        # Check that all agent_net_keys are in trainer_networks
        assert unique_net_keys == unique_trainer_net_keys
        # Setup specs for each network
        self._net_spec_keys = {}
        for i in range(len(unique_net_keys)):
            self._net_spec_keys[unique_net_keys[i]] = agents[i % len(agents)]

        # Setup table_network_config
        table_network_config = {}
        for trainer_key in self._trainer_networks.keys():
            most_matches = 0
            trainer_nets = self._trainer_networks[trainer_key]
            for sample in self._network_sampling_setup:  # type: ignore
                matches = 0
                for entry in sample:
                    if entry in trainer_nets:
                        matches += 1
                if most_matches < matches:
                    matches = most_matches
                    table_network_config[trainer_key] = sample

        extra_specs = {}
        # if issubclass(executor_fn, executors.RecurrentExecutor):
        #     extra_specs = self._get_extra_specs()

        int_spec = specs.DiscreteArray(len(unique_net_keys))
        agents = environment_spec.get_agent_ids()
        net_spec = {"network_keys": {agent: int_spec for agent in agents}}
        extra_specs.update(net_spec)

    def on_building_distributor_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_distributor_tables(self, builder: SystemBuilder) -> None:
        """[summary]"""
        with builder.program.group("tables"):
            builder.program_tables = builder.program.add_node(
                lp.ReverbNode(builder.tables)
            )

    def on_building_distributor_variable_server(self, builder: SystemBuilder) -> None:
        """[summary]"""
        with builder.program.group("variable_server"):
            builder.program_variable_server = builder.program.add_node(
                lp.CourierNode(builder.variable_server)
            )

    def on_building_distributor_trainer(self, builder: SystemBuilder) -> None:
        """[summary]"""
        with builder.program.group("trainer"):
            # Add executors which pull round-robin from our variable sources.
            for trainer_id in range(len(builder.config.trainer_networks.keys())):
                builder.program.add_node(
                    lp.CourierNode(
                        builder.trainer,
                        trainer_id,
                        builder.program_tables,
                        builder.program_variable_server,
                    )
                )

    def on_building_distributor_evaluator(self, builder: SystemBuilder) -> None:
        """[summary]"""
        with builder.program.group("evaluator"):
            builder.program.add_node(
                lp.CourierNode(builder.evaluator, builder.program_variable_server)
            )

    def on_building_distributor_executor(self, builder: SystemBuilder) -> None:
        """[summary]"""
        with builder.program.group("executor"):
            # Add executors which pull round-robin from our variable sources.
            for executor_id in range(builder._num_exectors):
                builder.program.add_node(
                    lp.CourierNode(
                        builder.executor,
                        executor_id,
                        builder.program_tables,
                        builder.program_variable_server,
                    )
                )

    def on_building_distributor_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass
