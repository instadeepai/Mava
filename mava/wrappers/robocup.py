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

from typing import Tuple

import dm_env
import gym
import numpy as np
from acme import types

from mava.utils.environments.RoboCup_env.RoboCup2D_env import RoboCup2D
from mava.utils.environments.RoboCup_env.robocup_utils.util_functions import (  # type: ignore # noqa: E501
    SpecWrapper,
)


class RoboCupWrapper(SpecWrapper):
    """Environment wrapper for 2D RoboCup environment."""

    def __init__(self, environment: RoboCup2D) -> None:
        self._environment = environment
        self._reset_next_step = True
        assert environment.game_setting in ["reward_shaping", "domain_randomisation"]

        super().__init__(environment.num_players)

    def reset(self) -> Tuple[dm_env.TimeStep, np.array]:
        """Resets the episode."""
        self._reset_next_step = False
        raw_obs, _, state = self._environment.reset()
        proc_obs = self._proc_robocup_obs(observations=raw_obs, done=False)
        proccessed_state = self._proc_robocup_state(state, proc_obs)
        timestep = dm_env.restart(proc_obs)
        return timestep, {"env_state": proccessed_state}

    def step(self, nn_actions: types.NestedArray) -> Tuple[dm_env.TimeStep, np.array]:
        """Steps the environment."""
        if self._reset_next_step:
            return self.reset()

        actions = self._proc_robocup_actions(nn_actions)
        raw_obs, rewards, state, done = self._environment.step(actions)
        self._reset_next_step = done

        proc_obs = self._proc_robocup_obs(
            observations=raw_obs, done=done, nn_actions=nn_actions
        )
        proccessed_state = self._proc_robocup_state(state, proc_obs)

        if done:
            self._step_type = dm_env.StepType.LAST
        else:
            self._step_type = dm_env.StepType.MID

        return (
            dm_env.TimeStep(
                observation=proc_obs,
                reward=rewards,
                discount=self._discount,
                step_type=self._step_type,
            ),
            {"env_state": proccessed_state},
        )

    @property
    def environment(self) -> gym.Env:
        """Returns the wrapped environment."""
        return self._environment
