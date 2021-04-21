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

import copy
from typing import Tuple

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


class TwoStepEnv:
    def __init__(self) -> None:
        self.step_num = 0
        self.state = 0
        self.prev_state = 0
        self.env_done = False

    def step(self, actions: Tuple[int, int]) -> Tuple[int, int, bool]:
        self.prev_state = copy.deepcopy(self.state)
        self.env_done = True  # Assume state > 0

        if self.state == 0:
            self.env_done = False
            if actions[0] == 0:
                self.state = 1
                return 1, 0, self.env_done  # Go to 2A
            else:
                self.state = 2
                return 2, 0, self.env_done  # Go to 2B
        elif self.state == 1:  # State 2A
            return self.state, 7, self.env_done
        elif self.state == 2:  # State 2B
            if actions[0] == 0 and actions[1] == 0:
                reward = 0
            elif actions[0] == 0 and actions[1] == 1:
                reward = 1
            elif actions[0] == 1 and actions[1] == 0:
                reward = 1
            elif actions[0] == 1 and actions[1] == 1:
                reward = 8
            return self.state, reward, self.env_done
        else:
            raise Exception("invalid state:{}".format(self.state))

    def reset(self) -> int:
        self.state = 0
        return self.state
