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

# type: ignore

from typing import Dict, Optional

import dm_env
from acme import types

from mava.utils.environments.RoboCup_env.robocup_utils.game_object import Flag

true_flag_coords = Flag.FLAG_COORDS


class CustomExecutor:
    """A fixed executor to test in the environments are working."""

    def __init__(self, agents):
        # Convert action and observation specs.
        self.agents = agents

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {"": ()},
    ) -> None:
        pass

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Optional[Dict[str, types.NestedArray]] = {},
    ) -> None:
        pass

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Dict[str, types.NestedArray]:
        actions = {}
        for agent, observation in observations.items():
            # Pass the observation through the policy network.
            actions[agent] = self.agents[agent].get_action(observation.observation)

        # Return a numpy array with squeezed out batch dimension.
        return actions

    def update(self, wait: bool = False) -> None:
        pass
