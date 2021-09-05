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

from typing import List, Dict

from acme.specs import EnvironmentSpec

from mava.callbacks import Callback


class Trainer(Callback):
    def __init__(
        self,
        trainer_fn: Type[core.Executor],
        trainer_config: Dict[str, int],
    ):
        """[summary]

        Args:
            trainer_fn (Type[core.Executor]): [description]
            trainer_config (Dict[str, int]): [description]
        """
        self.trainer_fn = trainer_fn
        self.trainer_config = trainer_config

    def on_building_trainer(self, builder: SystemBuilder):
        # Convert network keys for the trainer.
        trainer_agents = self._agents[: len(self._trainer_table_entry)]
        trainer_agent_net_keys = {
            agent: self._trainer_table_entry[a_i]
            for a_i, agent in enumerate(trainer_agents)
        }

        # The learner updates the parameters (and initializes them).
        builder.trainer = self.trainer_fn(**self.trainer_config)