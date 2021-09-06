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

"""MADDPG system builder implementation."""

import copy

from acme.specs import EnvironmentSpec
from dm_env import specs as dm_specs

from mava import specs
from mava.systems.building import OnlineSystemBuilder
from mava.components import building

BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray

# construct default system components


class MADDPGBuilder(OnlineSystemBuilder):
    def convert_discrete_to_bounded(
        self, environment_spec: specs.MAEnvironmentSpec
    ) -> specs.MAEnvironmentSpec:
        """convert discrete action space to bounded continuous action space
        Args:
            environment_spec (specs.MAEnvironmentSpec): description of
                the action, observation spaces etc. for each agent in the system.
        Returns:
            specs.MAEnvironmentSpec: updated environment spec.
        """

        env_adder_spec: specs.MAEnvironmentSpec = copy.deepcopy(environment_spec)
        keys = env_adder_spec._keys
        for key in keys:
            agent_spec = env_adder_spec._specs[key]
            if type(agent_spec.actions) == DiscreteArray:
                num_actions = agent_spec.actions.num_values
                minimum = [float("-inf")] * num_actions
                maximum = [float("inf")] * num_actions
                new_act_spec = BoundedArray(
                    shape=(num_actions,),
                    minimum=minimum,
                    maximum=maximum,
                    dtype="float32",
                    name="actions",
                )

                env_adder_spec._specs[key] = EnvironmentSpec(
                    observations=agent_spec.observations,
                    actions=new_act_spec,
                    rewards=agent_spec.rewards,
                    discounts=agent_spec.discounts,
                )
        return env_adder_spec

        def on_building_make_replay_table_start(self):
            self._env_spec = self.convert_discrete_to_bounded(environment_spec)
