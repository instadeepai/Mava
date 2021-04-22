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

from typing import Any, Dict, Tuple, Union

import gym
import numpy as np
from gym import spaces

"""
A simple two-step cooperative matrix game for two agents. Adapted from
Qmix paper https://arxiv.org/abs/1803.11485 and tensorflow implementation
https://github.com/tocom242242/qmix_tf2/blob/master/two_step_env.py.
Actions:
0 or 1, corresponding to which matrix the agents choose for the next time
step.
"""

# NOTE (St John) I was in the process of making this a gym env. Will
# wait to see how we decide to structure a base wrapper class. Don't
# want to introduce tech debt by making this needlessly complex.


class TwoStepEnv(gym.Env):
    def __init__(self) -> None:

        self.num_agents = 2
        self.state = 0
        self.env_done = False

        self.action_spaces = {}
        self.observation_spaces = {}
        self.agent_ids = []

        for a_i in range(self.num_agents):
            agent_id = "agent_" + str(a_i)
            self.agent_ids.append(agent_id)
            self.action_spaces[agent_id] = spaces.Discrete(2)  # int64
            self.observation_spaces[agent_id] = spaces.Box(0, 1, shape=(1,))  # float32

        self.possible_agents = self.agent_ids

    def step(
        self, action_n: Dict[str, np.ndarray]
    ) -> Tuple[
        Dict[str, Union[np.array, Any]],
        Dict[str, Union[np.array, Any]],
        Dict[str, Union[bool, Any]],
        Dict[str, Dict[Any, Any]],
    ]:
        obs_n = {}
        reward_n = {}
        # done_n = {}

        if self.state == 0:
            self.env_done = False
            reward_n = {"agent_0": 0.0, "agent_1": 0.0}
            done_n = {"agent_0": False, "agent_1": False}
            if action_n["agent_0"] == 0:
                self.state = 1
                obs_n = {"agent_0": 1.0, "agent_1": 1.0}
                return obs_n, reward_n, done_n, {}  # Go to 2A
            else:
                self.state = 2
                obs_n = {"agent_0": 2.0, "agent_1": 2.0}
                return obs_n, reward_n, done_n, {}  # Go to 2B

        elif self.state == 1:  # State 2A
            self.env_done = True
            self.state = 0
            reward_n = {"agent_0": 7.0, "agent_1": 7.0}
            done_n = {"agent_0": True, "agent_1": True}
            return obs_n, reward_n, done_n, {}

        elif self.state == 2:  # State 2B
            self.env_done = True
            self.state = 0
            done_n = {"agent_0": True, "agent_1": True}
            if action_n["agent_0"] == 0 and action_n["agent_1"] == 0:
                reward_n = {"agent_0": 0.0, "agent_1": 0.0}
            elif action_n["agent_0"] == 0 and action_n["agent_1"] == 1:
                reward_n = {"agent_0": 1.0, "agent_1": 1.0}
            elif action_n["agent_0"] == 1 and action_n["agent_1"] == 0:
                reward_n = {"agent_0": 1.0, "agent_1": 1.0}
            elif action_n["agent_0"] == 1 and action_n["agent_1"] == 1:
                reward_n = {"agent_0": 8.0, "agent_1": 8.0}
            return obs_n, reward_n, done_n, {}

        else:
            raise Exception("invalid state:{}".format(self.state))

    def reset(self) -> Dict[str, np.array]:
        self.state = 0
        obs_n = {"agent_0": 0.0, "agent_1": 0.0}
        return obs_n
