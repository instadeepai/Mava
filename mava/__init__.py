# Copyright 2022 InstaDeep Ltd. All rights reserved.
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
__version__ = "0.2.0"

from gymnasium import register


register(
    id="Foraging-2s-15x15-4p-5f-coop",
    entry_point="lbforaging.foraging:ForagingEnv",
    kwargs={
        "players": 4,
        "min_player_level": 1,
        "max_player_level": 2,
        "field_size": (15, 15),
        "max_num_food": 5,
        "min_food_level": 1,
        "max_food_level": None,
        "sight": 2,
        "max_episode_steps": 100,
        "force_coop": True,
        "grid_observation": False,
    },
)
