# Adapted from https://arxiv.org/pdf/1605.06676.pdf.

from typing import Any, Callable, Dict, List, Tuple, Union

import gym
import numpy as np


class MultiAgentPrisonEnv(gym.Env):
    def __init__(
        self,
        num_agents: int = 3,
        reset_callback: Callable = None,
        reward_callback: Callable = None,
        observation_callback: Callable = None,
        info_callback: Callable = None,
        done_callback: Callable = None,
        shared_viewer: bool = True,
    ) -> None:

        self._num_agents = num_agents
        # Generate IDs and convert agent list to dictionary format.
        self.env_done = False
        self.agent_ids = []

        for a_i in range(self._num_agents):
            agent_id = "agent_" + str(a_i)
            self.agent_ids.append(agent_id)

        self.possible_agents = self.agent_ids
        # scenario callbacks
        self._reset_callback = reset_callback
        self._reward_callback = reward_callback
        self._observation_callback = observation_callback
        self._info_callback = info_callback
        self._done_callback = done_callback
        # environment parameters

        self.max_time = 4 * self._num_agents - 6

        self.selected_agent = -1

    def step(
        self, action_n: Dict[str, int]
    ) -> Tuple[
        Dict[str, Union[np.array, Any]],
        Union[dict, Dict[str, Union[float, Any]]],
        Dict[str, Any],
        Dict[str, dict],
    ]:
        obs_n = {}
        reward_n = {}
        done_n = {}

        # set action for interrogated agent
        selected_agent_id = self.agent_ids[self.selected_agent]
        action = action_n[selected_agent_id]
        # advance world state
        self.agent_history.append(self.selected_agent)
        self.seen_all = np.unique(self.agent_history).shape[0] == self._num_agents
        if action == 1:
            self.env_done = True
            self.tell = True

        self.selected_agent = np.random.randint(0, self._num_agents)
        self.time += 1
        if self.time >= self.max_time:
            self.env_done = True

        # record observation for each agent
        for a_i, agent_id in enumerate(self.agent_ids):
            obs_n[agent_id] = self._get_obs(a_i, agent_id)
            reward_n[agent_id] = self._get_reward(a_i, agent_id)
            done_n[agent_id] = self._get_done(agent_id)
            # info_n["n"][agent_id] = self._get_info(agent_id)

            if done_n[agent_id]:
                self.env_done = True

        return obs_n, reward_n, done_n, {}

    def reset(self) -> Dict[str, np.array]:
        # reset world
        if self._reset_callback is not None:
            self._reset_callback(self.world)
        else:
            self.agent_history: List[int] = []
            self.n_seen = 0
            self.time = 0
            self.tell = False
            self.selected_agent = np.random.randint(0, self._num_agents)

        self.env_done = False
        # record observations for each agent
        obs_n = {}
        for a_i, agent_id in enumerate(self.agent_ids):
            obs_n[agent_id] = self._get_obs(a_i, agent_id)
        return obs_n

    # get info used for benchmarking
    def _get_info(self, agent_id: str) -> Dict:
        if self._info_callback is None:
            return {}
        return self._info_callback(agent_id)

    # get observation for a particular agent
    def _get_obs(self, a_i: int, agent_id: str) -> np.array:
        if self._observation_callback is None:
            return 1 if a_i == self.selected_agent else 0
        return self._observation_callback(agent_id, a_i, self.world)

    # get dones for a particular agent
    # unused right now -- agents are allowed to go beyond the viewing screen
    def _get_done(self, agent_id: str) -> bool:
        if self._done_callback is None:
            return self.env_done
        done = self._done_callback(agent_id)
        return done

    # get reward for a particular agent
    def _get_reward(self, a_i: int, agent_id: str) -> float:
        if self._reward_callback is None:
            if not self.env_done:
                return 0.0
            return 1.0 if self.seen_all and self.tell else -1.0
        return self._reward_callback(agent_id, a_i, self.world)
