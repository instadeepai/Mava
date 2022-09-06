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

"""Custom components for MADQN system."""
import abc
from dataclasses import dataclass
from typing import Any

# Commmit
import numpy as np
from dm_env import specs

from mava.components.jax import Component
from mava.core_jax import SystemBuilder


@dataclass
class ExtrasActionInfoConfig:
    pass


class ExtrasSpec(Component):
    @abc.abstractmethod
    def __init__(self, config: Any) -> None:
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "extras_spec"


class ExtrasActionInfo(ExtrasSpec):
    def __init__(
        self,
        config: ExtrasActionInfoConfig = ExtrasActionInfoConfig(),
    ):
        """_summary_

        Args:
            config : _description_.
        """
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """[summary]"""
        # TODO: extras
        agent_specs = builder.store.ma_environment_spec.get_agent_environment_specs()
        builder.store.extras_specs = {"policy_info": {}}

        for agent, spec in agent_specs.items():
            # Make dummy log_probs
            builder.store.extras_specs["policy_info"][agent] = np.ones(
                shape=spec.actions.num_values, dtype=np.float32
            )

        # Add the networks keys to extras.
        int_spec = specs.DiscreteArray(len(builder.store.unique_net_keys))
        agents = builder.store.ma_environment_spec.get_agent_ids()
        net_spec = {"network_keys": {agent: int_spec for agent in agents}}
        builder.store.extras_specs.update(net_spec)
        # adding network keys to next_extras
        builder.store.next_extras_specs = net_spec  # this could be easily set to {}

    @staticmethod
    def name() -> str:
        """_summary_

        Returns:
            _description_
        """
        return "extras_spec"
