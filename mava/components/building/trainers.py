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

from typing import Dict, Any, List

from mava.core import SystemBuilder
from mava.systems.training import Trainer
from mava.callbacks import Callback


class Trainer(Callback):
    def __init__(
        self,
        config: Dict[str, Any],
        components: List[Callback],
    ):
        """[summary]

        Args:
            trainer (SystemTrainer): [description]
        """
        self.config = config
        self.components = components

    def on_building_trainer_logger(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # create trainer logger
        trainer_logger_config = {}
        if builder._logger_config and "trainer" in builder._logger_config:
            trainer_logger_config = builder._logger_config["trainer"]
        trainer_logger = builder._logger_factory(  # type: ignore
            builder._trainer_id, **trainer_logger_config
        )
        builder.trainer_logger = trainer_logger

    def on_building_trainer(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # create networks
        networks = builder.system()

        # create dataset
        dataset = builder.dataset(builder._replay_client, builder._trainer_id)

        # convert network keys for the trainer
        trainer_agents = builder._agents[: len(builder._trainer_table_entry)]
        trainer_agent_net_keys = {
            agent: builder._trainer_table_entry[a_i]
            for a_i, agent in enumerate(trainer_agents)
        }

        # update config
        self.config.update(
            {
                "networks": networks,
                "dataset": dataset,
                "agent_net_keys": trainer_agent_net_keys,
                "variable_client": builder.trainer_variable_client,
                "counts": builder.trainer_counts,
                "logger": builder.trainer_logger,
            }
        )

        # trainer
        builder.trainer = Trainer(self.config, self.components)

    def on_building_trainer_end(self, builder: SystemBuilder) -> None:
        """[summary]"""

        # NB If using both NetworkStatistics and TrainerStatistics, order is important.
        # NetworkStatistics needs to appear before TrainerStatistics.
        # TODO(Kale-ab/Arnu): need to fix wrapper type issues
        builder.trainer = NetworkStatisticsActorCritic(builder.trainer)  # type: ignore

        builder.trainer = ScaledDetailedTrainerStatistics(  # type: ignore
            builder.trainer, metrics=["policy_loss", "critic_loss"]
        )