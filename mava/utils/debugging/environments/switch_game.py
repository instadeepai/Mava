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


from typing import Any, Dict, List, Tuple, Union

import gym
import numpy as np
from gym import spaces

"""
DIAL switch game implementation
Adapted from https://arxiv.org/pdf/1605.06676.pdf.

Actions:
0 - Nothing
1 - Tell
"""


class MultiAgentSwitchGame(gym.Env):
    def __init__(
        self,
        num_agents: int = 3,
    ) -> None:

        self.num_agents = num_agents
        # Generate IDs and convert agent list to dictionary format.
        self.env_done = False
        self.agent_ids = []

        # spaces
        self.action_spaces = {}
        self.observation_spaces = {}
        for a_i in range(self.num_agents):
            agent_id = "agent_" + str(a_i)
            self.agent_ids.append(agent_id)
            self.action_spaces[agent_id] = spaces.Discrete(2)
            self.observation_spaces[agent_id] = spaces.Box(
                low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32
            )

        self.possible_agents = self.agent_ids

        # environment parameters
        self.max_time = 4 * self.num_agents - 6
        self.selected_agent = -1

    def step(
        self, action_n: Dict[str, int]
    ) -> Tuple[
        Dict[str, Union[np.array, Any]],
        Union[dict, Dict[str, Union[float, Any]]],
        Dict[str, Any],
        Dict[str, dict],
    ]:
        obs_n = {}
        reward_n = {}
        done_n = {}

        # set action for interrogated agent
        selected_agent_id = self.agent_ids[self.selected_agent]

        # advance world state
        self.agent_history.append(self.selected_agent)
        self.seen_all = np.unique(self.agent_history).shape[0] == self.num_agents
        if action_n[selected_agent_id] == 1:
            self.env_done = True
            self.tell = True

        self.time += 1

        if self.time >= self.max_time:
            self.env_done = True
        else:
            self.selected_agent = self._agent_order[self.time]

        # record observation for each agent
        for a_i, agent_id in enumerate(self.agent_ids):
            obs_n[agent_id] = self._get_obs(a_i, agent_id)
            reward_n[agent_id] = self._get_reward(a_i, agent_id)
            done_n[agent_id] = self._get_done(agent_id)

            if done_n[agent_id]:
                self.env_done = True

        return obs_n, reward_n, done_n, {}

    def reset(self) -> Dict[str, np.array]:
        # reset world
        self.agent_history: List[int] = []
        self.n_seen = 0
        self.time = 0
        self.tell = False

        self._agent_order = np.array([0])
        while len(np.unique(self._agent_order)) < self.num_agents:
            self._agent_order = np.random.randint(0, self.num_agents, (self.max_time,))

        self.selected_agent = self._agent_order[0]

        self.env_done = False
        # record observations for each agent
        obs_n = {}
        for a_i, agent_id in enumerate(self.agent_ids):
            obs_n[agent_id] = self._get_obs(a_i, agent_id)
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent_id: str) -> Dict:
        return {}

    # get observation for a particular agent
    def _get_obs(self, a_i: int, agent_id: str) -> np.array:
        selected = 1.0 if a_i == self.selected_agent else 0.0
        return np.array([selected, a_i], dtype=np.float32)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent_id: str) -> bool:
        return self.env_done

    # get reward for a particular agent
    def _get_reward(self, a_i: int, agent_id: str) -> float:
        if not self.env_done:
            return np.array(0.0, dtype=np.float32)
        return (
            np.array(1.0, dtype=np.float32)
            if self.seen_all and self.tell
            else np.array(-1.0, dtype=np.float32)
        )
