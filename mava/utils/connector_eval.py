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

# Auto-generated level data
# Do not edit manually

from typing import List

import chex
import jax
import jax.numpy as jnp
from jumanji.environments.routing.connector import Connector
from jumanji.environments.routing.connector.constants import POSITION, TARGET
from jumanji.environments.routing.connector.generator import Generator
from jumanji.environments.routing.connector.types import Agent, State

LEVELS = {
    "cc_easy_5_3_2": {
        "grid_size": 5,
        "num_agents": 3,
        "start_pos": [[2, 3], [3, 1], [4, 0]],
        "target_pos": [[3, 2], [2, 2], [0, 0]],
    },
    "cc_med_9_7": {
        "grid_size": 9,
        "num_agents": 7,
        "start_pos": [[1, 0], [2, 0], [4, 0], [5, 7], [4, 4], [1, 5], [1, 7]],
        "target_pos": [[0, 3], [7, 2], [7, 7], [3, 4], [6, 3], [3, 6], [5, 5]],
    },
    "cc_hard_5_5_0": {
        "grid_size": 5,
        "num_agents": 5,
        "start_pos": [[1, 3], [2, 2], [1, 4], [1, 2], [4, 0]],
        "target_pos": [[3, 3], [0, 4], [4, 3], [2, 1], [4, 2]],
    },
    "cc_hard_9_9": {
        "grid_size": 9,
        "num_agents": 9,
        "start_pos": [[1, 0], [2, 1], [2, 0], [4, 0], [5, 7], [4, 4], [1, 5], [1, 7], [2, 8]],
        "target_pos": [[0, 3], [1, 8], [7, 2], [7, 7], [3, 4], [6, 3], [3, 6], [5, 5], [8, 4]],
    },
    "test_level_0": {
        "grid_size": 10,
        "num_agents": 10,
        "start_pos": [
            [2, 1],
            [8, 1],
            [0, 1],
            [3, 9],
            [2, 9],
            [7, 4],
            [6, 5],
            [9, 3],
            [0, 6],
            [1, 4],
        ],
        "target_pos": [
            [7, 1],
            [1, 1],
            [9, 1],
            [4, 9],
            [5, 9],
            [7, 6],
            [8, 5],
            [9, 7],
            [0, 4],
            [1, 6],
        ],
    },
    "cc_med_7_0": {
        "grid_size": 7,
        "num_agents": 5,
        "start_pos": [[2, 2], [6, 6], [5, 6], [5, 0], [6, 0]],
        "target_pos": [[4, 2], [5, 2], [2, 4], [3, 4], [3, 5]],
    },
    "cc_hard_5_5_1": {
        "grid_size": 5,
        "num_agents": 5,
        "start_pos": [[0, 2], [4, 1], [3, 1], [3, 4], [4, 4]],
        "target_pos": [[4, 2], [1, 0], [0, 0], [0, 3], [1, 3]],
    },
    "cc_easy_5_3_1": {
        "grid_size": 5,
        "num_agents": 3,
        "start_pos": [[0, 2], [4, 1], [3, 1]],
        "target_pos": [[4, 2], [1, 0], [0, 0]],
    },
    "cc_easy_5_3_0": {
        "grid_size": 5,
        "num_agents": 3,
        "start_pos": [[1, 3], [2, 2], [1, 4]],
        "target_pos": [[3, 3], [0, 4], [4, 3]],
    },
    "cc_hard_5_5_2": {
        "grid_size": 5,
        "num_agents": 5,
        "start_pos": [[2, 3], [3, 1], [4, 0], [1, 0], [0, 4]],
        "target_pos": [[3, 2], [2, 2], [0, 0], [3, 0], [0, 2]],
    },
    "cc_easy_7_4_0": {
        "grid_size": 7,
        "num_agents": 4,
        "start_pos": [[2, 2], [6, 6], [5, 6], [5, 0]],
        "target_pos": [[4, 2], [5, 2], [2, 4], [3, 4]],
    },
    "cc_easy_9_5": {
        "grid_size": 9,
        "num_agents": 5,
        "start_pos": [[1, 0], [2, 0], [4, 0], [5, 7], [4, 4]],
        "target_pos": [[0, 3], [7, 2], [7, 7], [3, 4], [6, 3]],
    },
    "cc_easy_5_3": {
        "grid_size": 5,
        "num_agents": 3,
        "start_pos": [[0, 0], [4, 4], [4, 3]],
        "target_pos": [[2, 4], [3, 2], [3, 1]],
    },
    "cc_hard_5_5": {
        "grid_size": 5,
        "num_agents": 5,
        "start_pos": [[0, 0], [4, 4], [4, 3], [1, 3], [2, 3]],
        "target_pos": [[2, 4], [3, 2], [3, 1], [4, 0], [2, 1]],
    },
}


class SingletonGenerator(Generator):
    def __init__(self, level_name: str):
        self._grid_size = LEVELS[level_name]["grid_size"]
        self._num_agents = LEVELS[level_name]["num_agents"]
        level_data = LEVELS[level_name]

        grid = jnp.zeros((self._grid_size, self._grid_size), dtype=jnp.int32)
        start_pos = jnp.array(level_data["start_pos"])
        target_pos = jnp.array(level_data["target_pos"])
        grid = grid.at[start_pos[:, 0], start_pos[:, 1]].set(
            POSITION + jnp.arange(self._num_agents) * 3
        )
        grid = grid.at[target_pos[:, 0], target_pos[:, 1]].set(
            TARGET + jnp.arange(self._num_agents) * 3
        )

        agents = Agent(
            id=jnp.arange(self._num_agents), start=start_pos, target=target_pos, position=start_pos
        )
        self.state = State(
            grid=grid,
            step_count=jnp.array(0),
            agents=agents,
            key=jax.random.PRNGKey(0),
        )

    def __call__(self, key: chex.PRNGKey) -> State:
        return self.state


def get_eval_envs() -> List[Connector]:
    envs = []
    for level_name in LEVELS.keys():
        generator = SingletonGenerator(level_name)
        envs.append(Connector(generator=generator))

    return envs
