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

"""Custom components for IPPO system."""
from types import SimpleNamespace

from mava.components.building.extras_spec import ExtrasSpec
from mava.core_jax import SystemBuilder


class DQNExtrasSpec(ExtrasSpec):
    def __init__(
        self,
        config: SimpleNamespace = SimpleNamespace(),
    ):
        """Class that adds log probs to the extras spec

        Args:
            config : SimpleNamespace
        """
        self.config = config

    def on_building_init_end(self, builder: SystemBuilder) -> None:
        """Create extra specs after builder has been initialised

        Args:
            builder: SystemBuilder

        Returns:
            None.
        """
        # builder.store.extras_spec = {}
        #
        # # Add the networks keys to extras.
        # int_spec = specs.DiscreteArray(len(builder.store.unique_net_keys))
        # agents = builder.store.ma_environment_spec.get_agent_ids()
        # net_spec = {"network_keys": {agent: int_spec for agent in agents}}

        # builder.store.extras_spec.update(net_spec)

        builder.store.extras_spec = self.get_network_keys(
            builder.store.unique_net_keys,
            builder.store.ma_environment_spec.get_agent_ids(),
        )
