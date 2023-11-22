from typing import Dict, List, Tuple

import gym
import numpy as np
import rware
from chex import Array
from gym.spaces import Box, MultiDiscrete


class RwareGymWrapper(gym.Wrapper):
    """Environment wrapper RWARE."""

    def __init__(
        self,
        env: rware.warehouse.Warehouse,
        team_reward: bool = False,
    ):
        super().__init__(env)
        self.team_reward = team_reward
        self.num_agents = self.env.n_agents
        self.num_actions = self.env.action_space[0].n
        self.obs_dim = self.env.observation_space[0].shape[0]
        self.metadata = self.env.metadata

        _obs_shape = (self.num_agents, self.obs_dim)
        _obs_low, _obs_high, _obs_dtype = (-np.inf, np.inf, self.env.observation_space[0].dtype)
        self.observation_space = Box(
            low=_obs_low, high=_obs_high, shape=_obs_shape, dtype=_obs_dtype
        )
        self.action_space = MultiDiscrete(
            nvec=[self.num_actions] * self.num_agents,
            dtype=np.int32,
        )

    def reset(self) -> Tuple[Array, Array, Array, Dict]:
        """Resets the env."""

        # Reset the environment
        observations = self.env.reset()

        observations = np.array(observations)

        return observations

    def step(self, actions: List):
        """Steps in env."""

        # Step the RWARE environment
        next_observation, reward, done, info = self.env.step(actions)

        if self.team_reward:
            reward = np.mean(reward)
        else:
            reward = np.array(reward)

        return np.array(next_observation), reward, done, info


def make_env(
    task_name: str = "rware-tiny-2ag-v1",
    team_reward: bool = True,
):
    def thunk():
        env = RwareGymWrapper(
            env=gym.make(task_name),
            team_reward=team_reward,
        )

        return env

    return thunk
