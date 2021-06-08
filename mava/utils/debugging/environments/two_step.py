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

"""
A simple two-step cooperative matrix game for two agents. Adapted from
Qmix paper https://arxiv.org/abs/1803.11485 and tensorflow implementation
https://github.com/tocom242242/qmix_tf2/blob/master/two_step_env.py.

Actions:
0 or 1 - corresponds with which matrix the agents choose for the next time
step.

Observations:
0, 1 or 2 - corresponds with which state the system of agents is in. Both
agents in the system will always have the same state. System starts in
state 0 and moves to state 1 or state 2 depending on the actions of agent
1 in the first timestep. System state resets after both agents act in
timestep 2.
"""


class TwoStepEnv(gym.Env):
    def __init__(self) -> None:
        self.num_agents = 2
        self.reset()
        self.agent_ids = []

        for a_i in range(self.num_agents):
            agent_id = "agent_" + str(a_i)
            self.agent_ids.append(agent_id)

        self.possible_agents = self.agent_ids

    def reset(
        self,
    ) -> Tuple[Dict[str, Union[np.array, Any]], Dict[str, Any]]:
        self.state = 0
        self.reward_n = {
            "agent_0": np.array(0.0, dtype=np.float32),
            "agent_1": np.array(0.0, dtype=np.float32),
        }
        self.env_done = False
        self.done_n = {"agent_0": False, "agent_1": False}
        self.obs_n = {
            "agent_0": np.array([0.0], dtype=np.float32),
            "agent_1": np.array([0.0], dtype=np.float32),
        }
        return self.obs_n, {"s_t": np.array(self.state, dtype=np.int64)}

    def step(
        self, action_n: Dict[str, int]
    ) -> Tuple[
        Dict[str, Union[np.array, Any]],
        Union[dict, Dict[str, Union[float, Any]]],
        Dict[str, Any],
        Dict[str, Any],
    ]:
        if self.state == 0:
            if action_n["agent_0"] == 0:
                self.state = 1
                self.obs_n = {
                    "agent_0": np.array([1.0], dtype=np.float32),
                    "agent_1": np.array([1.0], dtype=np.float32),
                }
                return (
                    self.obs_n,
                    self.reward_n,
                    self.done_n,
                    {"s_t": np.array(self.state, dtype=np.int64)},
                )  # Go to 2A
            else:
                self.state = 2
                self.obs_n = {
                    "agent_0": np.array([2.0], dtype=np.float32),
                    "agent_1": np.array([2.0], dtype=np.float32),
                }
                return (
                    self.obs_n,
                    self.reward_n,
                    self.done_n,
                    {"s_t": np.array(self.state, dtype=np.int64)},
                )  # Go to 2B

        elif self.state == 1:  # State 2A
            self.env_done = True
            self.state = 0
            self.reward_n = {
                "agent_0": np.array(7.0, dtype=np.float32),
                "agent_1": np.array(7.0, dtype=np.float32),
            }
            self.done_n = {"agent_0": True, "agent_1": True}
            return (
                self.obs_n,
                self.reward_n,
                self.done_n,
                {"s_t": np.array(self.state, dtype=np.int64)},
            )

        elif self.state == 2:  # State 2B
            self.env_done = True
            self.state = 0
            self.done_n = {"agent_0": True, "agent_1": True}
            if action_n["agent_0"] == 0 and action_n["agent_1"] == 0:
                self.reward_n = {
                    "agent_0": np.array(0.0, dtype=np.float32),
                    "agent_1": np.array(0.0, dtype=np.float32),
                }
            elif action_n["agent_0"] == 0 and action_n["agent_1"] == 1:
                self.reward_n = {
                    "agent_0": np.array(1.0, dtype=np.float32),
                    "agent_1": np.array(1.0, dtype=np.float32),
                }
            elif action_n["agent_0"] == 1 and action_n["agent_1"] == 0:
                self.reward_n = {
                    "agent_0": np.array(1.0, dtype=np.float32),
                    "agent_1": np.array(1.0, dtype=np.float32),
                }
            elif action_n["agent_0"] == 1 and action_n["agent_1"] == 1:
                self.reward_n = {
                    "agent_0": np.array(8.0, dtype=np.float32),
                    "agent_1": np.array(8.0, dtype=np.float32),
                }
            return (
                self.obs_n,
                self.reward_n,
                self.done_n,
                {"s_t": np.array(self.state, dtype=np.int64)},
            )

        else:
            raise Exception("invalid state:{}".format(self.state))
