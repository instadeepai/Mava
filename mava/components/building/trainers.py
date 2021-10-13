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

"""Trainer components for system builders"""

from typing import Dict, Any

from mava.systems.tf import variable_utils
from mava.callbacks import Callback
from mava.systems.building import SystemBuilder


class Trainer(Callback):
    def __init__(
        self,
        config: Dict[str, Any],
    ):
        """[summary]

        Args:
            config (Dict[str, Any]): [description]
            executor_fn (Type[core.Executor]): [description]
        """
        self.config = config

    def on_building_trainer_start(self, builder: SystemBuilder) -> None:
        """[summary]"""

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # create trainer logger
        trainer_logger_config = {}
        if builder._logger_config and "trainer" in builder._logger_config:
            trainer_logger_config = builder._logger_config["trainer"]
        trainer_logger = builder._logger_factory(  # type: ignore
            builder._trainer_id, **trainer_logger_config
        )
        builder._trainer_logger = trainer_logger

    def on_building_trainer(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # Create the system
        networks = builder.create_system()

        dataset = builder.dataset(builder._replay_client, builder._trainer_id)

        # This assumes agents are sort_str_num in the other methods
        agent_types = builder._agent_types
        max_gradient_norm = builder._config.max_gradient_norm
        discount = builder._config.discount
        target_update_period = builder._config.target_update_period
        target_averaging = builder._config.target_averaging
        target_update_rate = builder._config.target_update_rate

        # Create variable client
        variables = {}
        set_keys = []
        get_keys = []
        # TODO (dries): Only add the networks this trainer is working with.
        # Not all of them.
        for net_type_key in networks.keys():
            for net_key in networks[net_type_key].keys():
                variables[f"{net_key}_{net_type_key}"] = networks[net_type_key][
                    net_key
                ].variables
                if net_key in set(builder._trainer_networks):
                    set_keys.append(f"{net_key}_{net_type_key}")
                else:
                    get_keys.append(f"{net_key}_{net_type_key}")

        variables = builder.create_counter_variables(variables)
        count_names = [
            "trainer_steps",
            "trainer_walltime",
            "evaluator_steps",
            "evaluator_episodes",
            "executor_episodes",
            "executor_steps",
        ]
        get_keys.extend(count_names)
        counts = {name: variables[name] for name in count_names}

        variable_client = variable_utils.VariableClient(
            client=builder._variable_source,
            variables=variables,
            get_keys=get_keys,
            set_keys=set_keys,
            update_period=10,
        )

        # Get all the initial variables
        variable_client.get_all_and_wait()

        # Convert network keys for the trainer.
        trainer_agents = builder._agents[: len(builder._trainer_table_entry)]
        trainer_agent_net_keys = {
            agent: builder._trainer_table_entry[a_i]
            for a_i, agent in enumerate(trainer_agents)
        }

        trainer_config: Dict[str, Any] = {
            "agents": trainer_agents,
            "agent_types": agent_types,
            "policy_networks": networks["policies"],
            "critic_networks": networks["critics"],
            "observation_networks": networks["observations"],
            "target_policy_networks": networks["target_policies"],
            "target_critic_networks": networks["target_critics"],
            "target_observation_networks": networks["target_observations"],
            "agent_net_keys": trainer_agent_net_keys,
            "policy_optimizer": builder._config.policy_optimizer,
            "critic_optimizer": builder._config.critic_optimizer,
            "max_gradient_norm": max_gradient_norm,
            "discount": discount,
            "target_averaging": target_averaging,
            "target_update_period": target_update_period,
            "target_update_rate": target_update_rate,
            "variable_client": variable_client,
            "dataset": dataset,
            "counts": counts,
            "logger": builder._trainer_logger,
        }
        # if connection_spec:
        #     trainer_config["connection_spec"] = connection_spec

        # if issubclass(self._trainer_fn, training.MADDPGBaseRecurrentTrainer):
        #     trainer_config["bootstrap_n"] = self._config.bootstrap_n

        # The learner updates the parameters (and initializes them).
        trainer = builder._trainer_fn(**trainer_config)

        # NB If using both NetworkStatistics and TrainerStatistics, order is important.
        # NetworkStatistics needs to appear before TrainerStatistics.
        # TODO(Kale-ab/Arnu): need to fix wrapper type issues
        trainer = NetworkStatisticsActorCritic(trainer)  # type: ignore

        trainer = ScaledDetailedTrainerStatistics(  # type: ignore
            trainer, metrics=["policy_loss", "critic_loss"]
        )

        builder.trainer = trainer

    def on_building_trainer_end(self, builder: SystemBuilder) -> None:
        """[summary]"""