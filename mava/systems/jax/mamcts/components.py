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

"""Custom components for MAMCTS system."""
from dataclasses import dataclass

import numpy as np
from dm_env import specs

from mava.components.jax import Component
from mava.core_jax import SystemBuilder


# TODO remove mcts hardcodedness - sasha is a knob
@dataclass
class ExtraSearchPolicySpecConfig:
    num_actions: int = 5


class ExtraSearchPolicySpec(Component):
    def __init__(
        self,
        config: ExtraSearchPolicySpecConfig = ExtraSearchPolicySpecConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        agent_specs = builder.store.environment_spec.get_agent_specs()
        builder.store.extras_spec = {"policy_info": {}}

        for agent, spec in agent_specs.items():
            # Make dummy log_probs
            builder.store.extras_spec["policy_info"][agent] = np.ones(
                shape=(self.config.num_actions), dtype=np.float32
            )

        # Add the networks keys to extras.
        int_spec = specs.DiscreteArray(len(builder.store.unique_net_keys))
        agents = builder.store.environment_spec.get_agent_ids()
        net_spec = {"network_keys": {agent: int_spec for agent in agents}}
        builder.store.extras_spec.update(net_spec)

    @property
    def name(self) -> str:
        """_summary_

        Returns:
            _description_
        """
        return "extras_search_policy"
