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
from mava.systems.building import SystemBuilder
from mava.utils import enums

from mava.utils.sort_utils import sort_str_num, sample_new_agent_keys


class Distributor(Callback):
    def __init__(
        self,
        num_executors: int = 1,
        trainer_networks: Union[
            Dict[str, List], enums.Trainer
        ] = enums.Trainer.single_trainer,
        network_sampling_setup: Union[
            List, enums.NetworkSampler
        ] = enums.NetworkSampler.fixed_agent_networks,
        prefetch_size: int = 4,
        executor_variable_update_period: int = 1000,
        samples_per_insert: Optional[float] = 32.0,
        termination_condition: Optional[Dict[str, int]] = None,
    ) -> None:
        """[summary]

        Args:
            num_executors (int, optional): [description]. Defaults to 1.
            trainer_networks (Union[ Dict[str, List], enums.Trainer ], optional): [description]. Defaults to enums.Trainer.single_trainer.
            network_sampling_setup (Union[ List, enums.NetworkSampler ], optional): [description]. Defaults to enums.NetworkSampler.fixed_agent_networks.
            prefetch_size (int, optional): [description]. Defaults to 4.
            executor_variable_update_period (int, optional): [description]. Defaults to 1000.
            samples_per_insert (Optional[float], optional): [description]. Defaults to 32.0.
            termination_condition (Optional[Dict[str, int]], optional): [description]. Defaults to None.
        """

    def on_building_init(self, builder: SystemBuilder) -> None:
        # Setup agent networks and executor sampler
        agents = sort_str_num(builder.environment_spec.get_agent_ids())
        self._executor_samples = builder._system_config["executor_samples"]
        if not self._executor_samples:
            # if no executor samples provided, use shared_weights to determine setup
            self._agent_net_keys = {
                agent: agent.split("_")[0]
                if builder._system_config["shared_weights"]
                else agent
                for agent in agents
            }
            self._executor_samples = [
                [
                    self._agent_net_keys[key]
                    for key in sort_str_num(self._agent_net_keys.keys())
                ]
            ]
        else:
            # if executor samples provided, use executor_samples to determine setup
            _, self._agent_net_keys = sample_new_agent_keys(
                agents,
                self._executor_samples,
            )

        # Check that the environment and agent_net_keys has the same amount of agents
        sample_length = len(self._executor_samples[0])
        assert len(builder._environment_spec.get_agent_ids()) == len(
            self._agent_net_keys.keys()
        )

        # Check if the samples are of the same length and that they perfectly fit
        # into the total number of agents
        assert len(self._agent_net_keys.keys()) % sample_length == 0
        for i in range(1, len(self._executor_samples)):
            assert len(self._executor_samples[i]) == sample_length

        # Get all the unique agent network keys
        all_samples = []
        for sample in self._executor_samples:
            all_samples.extend(sample)
        unique_net_keys = sort_str_num(list(set(all_samples)))

        # Create mapping from ints to networks
        builder._net_to_ints = {net_key: i for i, net_key in enumerate(unique_net_keys)}

        # Setup trainer_networks
        if not builder._system_config["trainer_networks"]:
            self._trainer_networks = {"trainer_0": list(unique_net_keys)}
        else:
            self._trainer_networks = builder._system_config["trainer_networks"]

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
        for t_id in range(len(self._trainer_networks.keys())):
            most_matches = 0
            trainer_nets = self._trainer_networks[f"trainer_{t_id}"]
            for sample in self._executor_samples:
                matches = 0
                for entry in sample:
                    if entry in trainer_nets:
                        matches += 1
                if most_matches < matches:
                    matches = most_matches
                    table_network_config[f"trainer_{t_id}"] = sample

        self._table_network_config = table_network_config

        builder._extra_specs = {}

        int_spec = specs.DiscreteArray(len(unique_net_keys))
        agents = builder._environment_spec.get_agent_ids()
        net_spec = {"network_keys": {agent: int_spec for agent in agents}}
        builder._extra_specs.update(net_spec)

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
