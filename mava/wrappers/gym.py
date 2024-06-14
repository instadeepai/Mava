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

from typing import Dict, Tuple

import gym
import numpy as np
from numpy.typing import NDArray

from mava.types import Observation


class GymWrapper(gym.Wrapper):
    """Wrapper for gym environments"""

    def __init__(
        self,
        env: gym.env,
        use_individual_rewards: bool = False,
        add_global_state: bool = False,
        eval_env: bool = False,
    ):
        """Initialize the gym wrapper

        Args:
            env (gym.env): gym env instance.
            use_individual_rewards (bool, optional): Use individual or group rewards.
            Defaults to False.
            add_global_state (bool, optional) : Create global observations. Defaults to False.
            eval_env (bool, optional): Weather the instance is used for training or evaluation.
            Defaults to False.
        """
        super().__init__(env)
        self._env = env
        self.use_individual_rewards = use_individual_rewards
        self.add_global_state = add_global_state  # todo : add the global observations
        self.eval_env = eval_env
        self.num_agents = self._env.n_agents
        self.num_actions = self._env.action_space[
            0
        ].n  # todo: all the agents must have the same num_actions, add assertion?
        self.step_count = 0  # todo : make sure this implementaion is correct

    def reset(self) -> Tuple:
        agents_view, extra = self._env.reset(
            seed=np.random.randint(1), option={}
        )  # todo: assure reproducibility
        reward = np.zeros(self._env.n_agents)
        terminated, truncated = np.zeros(self._env.n_agents, dtype=bool), np.zeros(
            self._env.n_agents, dtype=bool
        )
        actions_mask = self._get_actions_mask(extra)

        obs = Observation(
            agents_view=np.array(agents_view), action_mask=actions_mask, step_count=self.step_count
        )

        return obs, reward, terminated, truncated, extra

    def step(self, actions: NDArray) -> Tuple:

        self.step_count += 1
        if (
            self._reset_next_step and not self.eval_env
        ):  # only auto-reset in training envs. todo: turn this into a sepreat wrapper
            return self.reset()

        agents_view, reward, terminated, truncated, extra = self.env.step(actions)

        terminated, truncated = np.array(terminated), np.array(truncated)

        done = np.logical_or(terminated, truncated).all()

        if (
            done and not self.eval_env
        ):  # only auto-reset in training envs, same functionality as the AutoResetWrapper.
            return self.reset()

        actions_mask = self._get_actions_mask(extra)

        if self.use_individual_rewards:
            reward = np.array(reward)
        else:
            reward = np.array([np.array(reward).mean()] * self.num_agents)

        obs = Observation(
            agents_view=np.array(agents_view), action_mask=actions_mask, step_count=self.step_count
        )

        return obs, reward, terminated, truncated, extra

    def _get_actions_mask(self, extra: Dict) -> NDArray:
        if "action_mask" in extra:
            return np.array(extra["action_mask"])
        return np.ones((self.num_agents, self.num_actions), dtype=np.float32)
