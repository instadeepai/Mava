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

# See SMAC here: https://github.com/oxwhirl/smac
# Documentation available at smac/blob/master/docs/smac.md

"""Wraps a StarCraft II MARL environment (SMAC) as a dm_env environment."""

from typing import Any, Dict, List, Tuple

import dm_env
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from gym import spaces
from gym.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv

try:
    from smac.env import StarCraft2Env
except ModuleNotFoundError:
    pass


from mava import types
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
from mava.wrappers.env_wrappers import ParallelEnvWrapper


class SMACEnvWrapper(ParallelEnvWrapper):
    """Wraps a StarCraft II MARL environment (SMAC) as a Mava Parallel environment.
    Based on RLlib & Pettingzoo wrapper provided by SMAC.
    Args:
        ParallelEnvWrapper ([type]): [description]
    """

    def __init__(self, environment: "StarCraft2Env") -> None:
        """Create a new multi-agent StarCraft env compatible with Mava.
        Args:
            environment (StarCraft2Env): Arguments to pass to the underlying
                smac.env.starcraft.StarCraft2Env instance.
        """

        self._environment = environment
        self.reset()

    def _get_agents(self) -> List:
        """Function that returns agent names and ids.
        Returns:
            List: list containing agents in format {agent_name}_{agent_id}.
        """
        agent_types = {
            self._environment.marine_id: "marine",
            self._environment.marauder_id: "marauder",
            self._environment.medivac_id: "medivac",
            self._environment.hydralisk_id: "hydralisk",
            self._environment.zergling_id: "zergling",
            self._environment.baneling_id: "baneling",
            self._environment.stalker_id: "stalker",
            self._environment.colossus_id: "colossus",
            self._environment.zealot_id: "zealot",
        }

        agents = []
        for agent_id, agent_info in self._environment.agents.items():
            agents.append(f"{agent_types[agent_info.unit_type]}_{agent_id}")
        return agents

    def _observe_all(self, obs_list: List) -> Dict:
        """Function that combibnes all agent observations into a single dict.
        Args:
            obs_list (List): list of all agent observations.
        Returns:
            Dict: dict containing agent observations and action masks.
        """
        observe = {}
        for i, obs in enumerate(obs_list):
            observe[self.possible_agents[i]] = {
                "observation": obs,
                "action_mask": (
                    np.array(self._environment.get_avail_agent_actions(i)).astype(bool)
                ).astype(int),
            }
        return observe

    def reset(self) -> Tuple[dm_env.TimeStep, np.array]:
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        self._env_done = False
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

        # reset internal SC2 env
        obs_list, state = self._environment.reset()

        # Initialize Spaces
        # Agents only become populated after reset
        self._possible_agents = self._get_agents()
        self._agents = self._possible_agents[:]

        self.action_spaces = {
            agent: Discrete(self._environment.get_total_actions())
            for agent in self._agents
        }
        self.observation_spaces = {
            agent: spaces.Dict(
                {
                    "observation": Box(
                        -1,
                        1,
                        shape=(self._environment.get_obs_size(),),
                        dtype="float32",
                    ),
                    "action_mask": Box(
                        0,
                        1,
                        shape=(self.action_spaces[agent].n,),
                        dtype=self.action_spaces[agent].dtype,
                    ),
                }
            )
            for agent in self._agents
        }

        # Convert observations
        observe = self._observe_all(obs_list)
        observations = self._convert_observations(
            observe, {agent: False for agent in self._possible_agents}
        )

        # create discount spec
        discount_spec = self.discount_spec()
        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self._possible_agents
        }

        # create rewards spec
        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self._possible_agents
        }

        # dm_env timestep
        timestep = parameterized_restart(rewards, self._discounts, observations)

        return timestep, {"s_t": state}

    def step(self, actions: Dict[str, np.ndarray]) -> Tuple[dm_env.TimeStep, np.array]:
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
        if self._reset_next_step:
            return self.reset()

        actions_feed = [actions[key] for key in self._agents]
        reward, terminated, _ = self._environment.step(actions_feed)
        obs_list = self._environment.get_obs()
        state = self._environment.get_state()
        self._env_done = terminated

        observe = self._observe_all(obs_list)
        dones = {agent: terminated for agent in self._possible_agents}

        observations = self._convert_observations(observe, dones)
        self._agents = list(observe.keys())
        rewards_spec = self.reward_spec()

        #  Handle empty rewards
        if not reward:
            rewards = {
                agent: convert_np_type(rewards_spec[agent].dtype, 0)
                for agent in self._possible_agents
            }
        else:
            rewards = {
                agent: convert_np_type(rewards_spec[agent].dtype, reward)
                for agent in self._agents
            }

        if self.env_done():
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
        else:
            self._step_type = dm_env.StepType.MID

        timestep = dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=self._discounts,
            step_type=self._step_type,
        )

        self.reward = rewards

        return timestep, {"s_t": state}

    def env_done(self) -> bool:
        """Returns a bool indicating if all agents in env are done.
        Returns:
            bool: Bool indicating if all agents are done.
        """
        return self._env_done

    def _convert_observations(
        self, observes: Dict[str, np.ndarray], dones: Dict[str, bool]
    ) -> types.Observation:
        """Converts observations to correct Mava format.
        Args:
            observes (Dict[str, np.ndarray]): Dict containing agent observations.
            dones (Dict[str, bool]): Dict indicating which agents are done.
        Returns:
            types.Observation: Correct format observations (OLT).
        """
        observations: Dict[str, types.OLT] = {}
        for agent, observation in observes.items():
            if isinstance(observation, dict) and "action_mask" in observation:
                legals = observation["action_mask"]
                observation = observation["observation"]
            else:
                legals = np.ones(
                    _convert_to_spec(self.action_space).shape,
                    dtype=self.action_space.dtype,
                )
            observations[agent] = types.OLT(
                observation=observation,
                legal_actions=legals,
                terminal=np.asarray([dones[agent]], dtype=np.float32),
            )

        return observations

    def observation_spec(self) -> types.Observation:
        """Function returns observation spec (format) of the env.
        Returns:
            types.Observation: Observation spec.
        """
        return {
            agent: types.OLT(
                observation=_convert_to_spec(
                    self.observation_spaces[agent]["observation"]
                ),
                legal_actions=_convert_to_spec(
                    self.observation_spaces[agent]["action_mask"]
                ),
                terminal=specs.Array((1,), np.float32),
            )
            for agent in self._possible_agents
        }

    def action_spec(self) -> Dict[str, specs.DiscreteArray]:
        """Function returns action spec (format) of the env.
        Returns:
            Dict[str, specs.DiscreteArray]: action spec.
        """
        return {
            agent: _convert_to_spec(self.action_spaces[agent])
            for agent in self._possible_agents
        }

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Function returns reward spec (format) of the env.
        Returns:
            Dict[str, specs.Array]: reward spec.
        """
        return {agent: specs.Array((), np.float32) for agent in self._possible_agents}

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns discount spec (format) of the env.
        Returns:
            Dict[str, specs.BoundedArray]: discount spec.
        """
        return {
            agent: specs.BoundedArray((), np.float32, minimum=0, maximum=1.0)
            for agent in self._possible_agents
        }

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns extra spec (format) of the env.
        Returns:
            Dict[str, specs.BoundedArray]: extra spec.
        """
        state = self._environment.get_state()
        # TODO (dries): What should the real bounds be of the state spec?
        return {
            "s_t": specs.BoundedArray(
                state.shape, np.float32, minimum=float("-inf"), maximum=float("inf")
            )
        }

    def seed(self, random_seed: int) -> None:
        """Function to seed the environment.
        Args:
            random_seed (int): random seed used when seeding the env.
        """
        self._environment._seed = random_seed
        # Reset after setting seed
        self.env.full_restart()

    @property
    def agents(self) -> List:
        """Returns active/not done agents in the env.
        Returns:
            List: active agents in the env.
        """
        return self._agents

    @property
    def possible_agents(self) -> List:
        """Returns all posible agents in the env.
        Returns:
            List: all possible agents in the env.
        """
        return self._possible_agents

    @property
    def environment(self) -> ParallelEnv:
        """Returns the wrapped environment."""
        return self._environment

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        return getattr(self._environment, name)
