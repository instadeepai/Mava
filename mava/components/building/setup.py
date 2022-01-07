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
import functools
from typing import Dict, Any

from dm_env import specs

from mava.utils.loggers import logger_utils
from mava.utils.sort_utils import sort_str_num, sample_new_agent_keys

from mava import specs as mava_specs
from mava.callbacks import Callback
from mava.systems.building import SystemBuilder


class Setup(Callback):
    def __init__(self, config: Dict[str, Dict[str, Any]]) -> None:
        """[summary]

        Args:
            config (Dict[str, Dict[str, Any]]): [description]
        """
        self.config = config

    def on_building_init_start(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_init(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        pass


class Default(Setup):
    def on_building_init_start(self, builder: SystemBuilder) -> None:

        builder._system_config = self.config["system"]
        builder._environment_spec = ["environment_spec"]
        builder._architecture_fn = builder._system_config["architecture"]
        builder._environment_factory = builder._system_config["environment_factory"]
        builder._network_factory = builder._system_config["network_factory"]
        builder._logger_factory = builder._system_config["logger_factory"]
        builder._num_executors = builder._system_config["num_executors"]
        builder._checkpoint_subpath = builder._system_config["checkpoint_subpath"]
        builder._checkpoint = builder._system_config["checkpoint"]
        builder._logger_config = builder._system_config["logger_config"]
        builder._train_loop_fn = builder._system_config["train_loop_fn"]
        builder._train_loop_fn_kwargs = builder._system_config["train_loop_fn_kwargs"]
        builder._eval_loop_fn = builder._system_config["eval_loop_fn"]
        builder._eval_loop_fn_kwargs = builder._system_config["eval_loop_fn_kwargs"]

        if not builder._environment_spec:
            builder.environment_spec = mava_specs.MAEnvironmentSpec(
                builder._environment_factory(evaluation=False)  # type: ignore
            )

        # set default logger if no logger provided
        if not builder._logger_factory:
            builder._logger_factory = functools.partial(
                logger_utils.make_logger,
                directory="~/mava",
                to_terminal=True,
                time_delta=10,
            )


class MultiTrainer(Default):
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
