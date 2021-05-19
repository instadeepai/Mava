# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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

# See SMAC here: https://github.com/oxwhirl/smac

"""Wraps a StarCraft II MARL environment (SMAC) as a dm_env environment."""
import random
from typing import Dict, List, Optional, Tuple, Type

import numpy as np
from gym.spaces import Box, Discrete
from smac.env import StarCraft2Env  # type:ignore

from mava.wrappers.env_wrappers import ParallelEnvWrapper  # , SequentialEnvWrapper


# Is it ParallelEnvWrapper or SequentialEnvWrapper
class SMACEnvWrapper(ParallelEnvWrapper):
    """
    Wraps a StarCraft II MARL environment (SMAC) as a Mava Parallel environment.
    Based on RLlib wrapper provided by SMAC.
    """

    def __init__(self, **smac_args: Optional[Tuple]) -> None:
        """Create a new multi-agent StarCraft env compatible with RLlib.
        Arguments:
            smac_args (dict): Arguments to pass to the underlying
                smac.env.starcraft.StarCraft2Env instance.
        """

        self._environment = StarCraft2Env(**smac_args)
        self._ready_agents: List = []
        self.observation_space = Dict(
            {
                "obs": Box(-1, 1, shape=(self._env.get_obs_size(),)),
                "action_mask": Box(0, 1, shape=(self._env.get_total_actions(),)),
            }
        )
        self.action_space: Type[Discrete] = Discrete(self._env.get_total_actions())

    def reset(self) -> Dict:
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        obs_list, state_list = self._env.reset()
        return_obs = {}
        for i, obs in enumerate(obs_list):
            return_obs[i] = {
                "action_mask": np.array(self._env.get_avail_agent_actions(i)),
                "obs": obs,
            }

        self._ready_agents = list(range(len(obs_list)))
        return return_obs  # TODO return global state for Mixers

    def step(self, action_dict: Dict) -> Tuple:
        """Returns observations from ready agents.
        The returns are dicts mapping from agent_id strings to values. The
        number of agents in the env can vary over time.
        Returns
        -------
            obs (dict): New observations for each ready agent.
            rewards (dict): Reward values for each ready agent. If the
                episode is just started, the value will be None.
            dones (dict): Done values for each ready agent. The special key
                "__all__" (required) is used to indicate env termination.
            infos (dict): Optional info values for each agent id.
        """

        actions = []
        for i in self._ready_agents:
            if i not in action_dict:
                raise ValueError(f"You must supply an action for agent: {i}")
            actions.append(action_dict[i])

        if len(actions) != len(self._ready_agents):
            raise ValueError(
                f"Number of actions ({len(actions)}) does not match number \
                    of ready agents (len(self._ready_agents))."
            )

        rew, done, info = self._env.step(actions)
        obs_list = self._env.get_obs()
        return_obs = {}
        for i, obs in enumerate(obs_list):
            return_obs[i] = {
                "action_mask": self._env.get_avail_agent_actions(i),
                "obs": obs,
            }
        rews = {i: rew / len(obs_list) for i in range(len(obs_list))}
        dones = {i: done for i in range(len(obs_list))}
        infos = {i: info for i in range(len(obs_list))}

        self._ready_agents = list(range(len(obs_list)))
        return return_obs, rews, dones, infos

    def env_done(self) -> bool:
        """
        Returns a bool indicating if all agents in env are done.
        """
        return self._environment.env_done  # TODO Check SMAC has this function

    @property
    def agents(self) -> List:
        """
        Returns the active agents in the env.
        """
        return NotImplementedError  # type:ignore

    @property
    def possible_agents(self) -> List:
        """
        Returns all the possible agents in the env.
        """
        return NotImplementedError  # type:ignore

    # Note sure we need these next methods. Comes from RLlib wrapper.
    def close(self) -> None:
        """Close the environment"""
        self._env.close()

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
