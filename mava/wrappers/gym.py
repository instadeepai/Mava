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


class GymRwareWrapper(gym.Wrapper):
    """Wrapper for rware gym environments"""

    def __init__(
        self,
        env: gym.Env,
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
        self._env = gym.wrappers.compatibility.EnvCompatibility(env)
        self.use_individual_rewards = use_individual_rewards
        self.add_global_state = add_global_state  # todo : add the global observations
        self.eval_env = eval_env
        self.num_agents = len(self._env.action_space)
        self.num_actions = self._env.action_space[
            0
        ].n  # todo: all the agents must have the same num_actions, add assertion?

    def reset(self) -> Tuple:
        (agents_view, info), _ = self._env.reset(
            seed=np.random.randint(1)
        )  # todo: assure reproducibility, this only works for rware

        info = {"action_mask" : self._get_actions_mask(info)}
        
        return np.array(agents_view), info

    def step(self, actions: NDArray) -> Tuple:

        agents_view, reward, terminated, truncated, info = self.env.step(actions)

        done = np.logical_or(terminated, truncated).all()

        if (
            done and not self.eval_env
        ):  # only auto-reset in training envs, same functionality as the AutoResetWrapper.
            agents_view, info = self.reset()
            reward = np.zeros(self.num_agents)
            terminated, truncated = np.zeros(self.num_agents, dtype=bool), np.zeros(
                self.num_agents, dtype=bool
            )
            return agents_view, reward, terminated, truncated, info

        info = {"action_mask" : self._get_actions_mask(info)}

        if self.use_individual_rewards:
            reward = np.array(reward)
        else:
            reward = np.array([np.array(reward).mean()] * self.num_agents)


        return agents_view, reward, terminated, truncated, info

    def _get_actions_mask(self, info: Dict) -> NDArray:
        if "action_mask" in info:
            return np.array(info["action_mask"])
        return np.ones((self.num_agents, self.num_actions), dtype=np.float32)
    

class AsyncGymWrapper(gym.Wrapper):
    """Wrapper for async gym environments"""

    def __init__(self, env: gym.vector.AsyncVectorEnv):
        super().__init__(env)
        self._env = env
        self.step_count = 0 #todo : make sure this is implemented correctly 
        
        action_space = env.single_action_space
        self.num_agents = len(action_space)
        self.num_actions = action_space[0].n 
        self.num_envs = env.num_envs
        
    def reset(self) -> Tuple[Observation, Dict]:
        agents_view , info = self._env.reset()
        obs = self._create_obs(agents_view, info)
        dones = np.zeros((self.num_envs, self.num_agents))
        return obs, dones, info

    def step(self, actions : NDArray) -> Tuple[Observation, NDArray, NDArray, NDArray, Dict]:
        
        self.step_count += 1
        actions = actions.swapaxes(0,1) # num_env, num_ags --> num_ags, num_env as expected by the async env
        agents_view, reward, terminated, truncated, info = self._env.step(actions)
        obs = self._create_obs(agents_view, info)
        dones = np.logical_or(terminated, truncated)
        return obs, reward, dones, info
        
         
    def _create_obs(self, agents_view : NDArray, info: Dict) -> Observation:
        """Create the observations"""
        agents_view = np.stack(agents_view, axis = 1)
        return Observation(agents_view=agents_view, action_mask=np.stack(info["action_mask"], axis = 0), step_count=self.step_count)
        