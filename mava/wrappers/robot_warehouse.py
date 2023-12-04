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

from typing import Callable, Dict, List, Tuple

import gym
import numpy as np
from chex import Array
from gym.spaces import Box, MultiDiscrete, flatdim


class GymWrapper(gym.Wrapper):
    """Environment wrapper for Robot-Warehouse."""

    def __init__(
        self,
        env: gym.Env,
        team_reward: bool = True,
    ):
        super().__init__(env)
        self.team_reward = team_reward
        self._agents = [f"agent_{n}" for n in range(self.env.n_agents)]
        self.num_agents = self.env.n_agents
        self.num_actions = self.env.action_space[0].n
        self._reset_next_step = True
        self._done = False
        self.max_episode_length = self.env.max_steps
        self.metadata = self.env._metadata

        # Define observation and action spaces
        _obs_shape = (self.num_agents, self.env.observation_space[0].shape[0])
        _obs_low, _obs_high, _obs_dtype = (
            self.env.observation_space[0].low[0],
            self.env.observation_space[0].high[0],
            np.float32,
        )
        self.observation_space = Box(
            low=_obs_low, high=_obs_high, shape=_obs_shape, dtype=_obs_dtype
        )
        self.action_space = MultiDiscrete(
            nvec=[self.num_actions] * self.num_agents,
            dtype=np.int32,
        )

        # TODO: fix legal actions
        self.longest_action_space = max(self.env.action_space, key=lambda x: x.n)

    def reset(self) -> Tuple[Array, Dict]:
        """Resets the env."""
        # Reset the environment
        observations, extra = self.env.reset()
        self._done = False
        self._reset_next_step = False

        # Get legal actions
        legal_actions = self._get_legal_actions()

        # Convert observations and legal actions to numpy arrays
        observations, legal_actions = self._convert_observations(observations, legal_actions)

        # TODO: use extra if needed
        info = {"extra_info": extra, "legal_actions": legal_actions}

        return observations, info

    def step(self, actions: List) -> Tuple[Array, float, bool, bool, Dict]:
        """Steps in env."""

        # Possibly reset the environment
        if self._reset_next_step:
            return self.reset()

        # Step the environment
        next_observations, reward, terminated, truncated, self._info = self.env.step(actions)

        # Check if env is done
        terminated = np.array(terminated).all()
        truncated = np.array(truncated).all()
        self._done = terminated or truncated
        if self._done:
            self._reset_next_step = True

        # Get legal actions
        legal_actions = self._get_legal_actions()

        # Convert observations and legal actions to numpy arrays
        next_observations, legal_actions = self._convert_observations(
            next_observations, legal_actions
        )

        # Reward info
        if self.team_reward:
            reward = np.array(np.array(reward).mean() * self.num_agents)
        else:
            reward = np.array(reward)

        # State info
        info = {"extra_info": self._info, "legal_actions": legal_actions}

        #TODO: return termination and truncation
        return next_observations, reward, self._done, False, info

    def _get_legal_actions(self) -> List:
        """Get legal actions from the environment."""
        legal_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            legal_actions.append(avail_agent)
        return legal_actions

    def get_avail_agent_actions(self, agent_id) -> List:
        """Returns the available actions for agent_id"""
        valid = flatdim(self.env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def _convert_observations(
        self,
        observations: List,
        legal_actions: List,
    ) -> Tuple[Array, Array]:
        """Convert observation so it's numpy array."""
        observations = np.array(observations, dtype="float32")
        legal_actions = np.array(legal_actions, dtype="float32")

        return observations, legal_actions


class AgentIDWrapper(gym.Wrapper):
    """Add onehot agent IDs to observation."""

    def __init__(self, env):
        super().__init__(env)

        self.agent_ids = np.eye(self.env.num_agents)

        _obs_dtype = self.env.observation_space.dtype
        _obs_shape = self.observation_space.shape
        _new_obs_shape = (self.env.num_agents, _obs_shape[1] + self.env.num_agents)
        self.observation_space = Box(low=-1, high=1, shape=_new_obs_shape, dtype=_obs_dtype)

    def reset(self) -> Tuple[Array, Dict]:
        """Reset the environment."""
        obs, info = self.env.reset()
        obs = np.concatenate([self.agent_ids, obs], axis=1)
        return obs, info

    def step(self, action: list) -> Tuple[Array, float, bool, bool, Dict]:
        """Step the environment."""
        #TODO: return termination and truncation
        obs, reward, done, _, info = self.env.step(action)
        obs = np.concatenate([self.agent_ids, obs], axis=1)
        return obs, reward, done, False, info


def make_env(
    map_name: str = "rware-tiny-2ag-v1",
    team_reward: bool = True,
    add_agent_id: bool = True,
) -> Callable:
    """Create a function which creates a fully configured environment."""

    def thunk() -> gym.Env:
        """Create an environment."""
        env = GymWrapper(
            env=gym.make(map_name),
            team_reward=team_reward,
        )
        if add_agent_id:
            env = AgentIDWrapper(env)
        return env

    return thunk
