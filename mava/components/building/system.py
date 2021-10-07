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


class System(Callback):
    def __init__(self, config: Dict[str, Dict[str, Any]]) -> None:
        """[summary]

        Args:
            config (Dict[str, Dict[str, Any]]): [description]
        """
        self.config = config

    def on_building_system_start(self, builder: SystemBuilder) -> None:
        """[summary]

        Args:
            builder (SystemBuilder): [description]
        """
        pass

    def on_building_system_networks(self, builder: SystemBuilder) -> None:
        """[summary]

        Args:
            builder (SystemBuilder): [description]
        """
        pass

    def on_building_system_architecture(self, builder: SystemBuilder) -> None:
        """[summary]

        Args:
            builder (SystemBuilder): [description]
        """
        pass

    def on_building_system(self, builder: SystemBuilder) -> None:
        """[summary]

        Args:
            builder (SystemBuilder): [description]
        """
        pass

    def on_building_system_end(self, builder: SystemBuilder) -> None:
        """[summary]

        Args:
            builder (SystemBuilder): [description]
        """
        pass


class Default(System):
    def on_building_system_networks(self, builder: SystemBuilder) -> None:
        # Create the networks to optimize (online)
        builder._networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
            net_spec_keys=self._net_spec_keys,
        )

    def on_building_system(self, builder: SystemBuilder) -> None:
        # TODO (dries): Can net_spec_keys and network_spec be used as
        # the same thing? Can we use use one of those two instead of both.
        builder._networks = builder._architecture.create_system()


class ActorCritic(Default):
    def on_building_system_architecture(self, builder: SystemBuilder) -> None:
        # architecture args
        architecture_config = {
            "environment_spec": builder._environment_spec,
            "observation_networks": builder._networks["observations"],
            "policy_networks": builder._networks["policies"],
            "critic_networks": builder._networks["critics"],
            "agent_net_keys": builder._agent_net_keys,
        }

        builder._architecture = builder._architecture_fn(**architecture_config)

    def on_building_system(self, builder: SystemBuilder) -> None:
        networks = builder._architecture.create_system()
        behaviour_networks = builder._architecture.create_behaviour_policy()
        builder.system_networks = (networks, behaviour_networks)