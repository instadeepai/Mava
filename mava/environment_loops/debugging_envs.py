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


"""A multi-agent/environment training loop for Petting Zoo."""
from typing import Union

import numpy as np

from mava.wrappers.debugging_envs import DebuggingEnvWrapper


def get_good_simple_spread_action(
    agent_id: int, obs: np.array, environment: DebuggingEnvWrapper
) -> Union[int, np.array]:
    import gym

    diff = np.array(obs[5:7])
    if type(environment.action_spaces[agent_id]) == gym.spaces.discrete.Discrete:
        x, y = diff
        if abs(x) > abs(y):
            if x < 0.0:
                return 1
            else:
                return 2
        else:
            if y < 0.0:
                return 3
            else:
                return 4
    elif type(environment.action_spaces[agent_id]) == gym.spaces.box.Box:
        return diff
    else:
        raise ValueError(
            "Unknown action space: ", type(environment.action_spaces[agent_id])
        )


def test_loop(environment: DebuggingEnvWrapper) -> None:
    while True:
        timestep = environment.reset()
        obs_n = timestep.observation
        tot_rewards = 0
        while not environment.env_done:
            action_n = {}
            for agent_id in environment.agent_ids:
                action_n[agent_id] = get_good_simple_spread_action(
                    agent_id, obs_n[agent_id].observation, environment
                )

            timestep = environment.step(action_n)
            obs_n = timestep.observation
            reward_n = timestep.reward

            tot_rewards += np.mean([r for r in reward_n.values()])
        print("Done. mean total rewards: ", tot_rewards)
