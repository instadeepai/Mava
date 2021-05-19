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
from typing import Any, Dict, List, Optional, Tuple, Type

import dm_env
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from gym.spaces import Box, Discrete
from pettingzoo.utils.env import ParallelEnv
from smac.env import StarCraft2Env  # type:ignore

from mava import types
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
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

        self._reset_next_step = True
        self._ready_agents: List = []
        self.observation_space = Dict(
            {
                "obs": Box(-1, 1, shape=(self._env.get_obs_size(),)),
                "action_mask": Box(0, 1, shape=(self._env.get_total_actions(),)),
            }
        )
        self.action_space: Type[Discrete] = Discrete(self._env.get_total_actions())

    def reset(self) -> Tuple[dm_env.TimeStep, np.array]:
        """Resets the env and returns observations from ready agents.
        Returns:
            obs (dict): New observations for each ready agent.
        """
        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

        # TODO Check the form of this state list and convert for return.
        obs_list, state_list = self._env.reset()
        observe: Dict[str, np.ndarray] = {}
        for i, obs in enumerate(obs_list):
            agent = f"agent_{i}"
            observe[agent] = {  # TODO Only obs in this Dict or mask too?
                "action_mask": np.array(self._env.get_avail_agent_actions(i)),
                "obs": obs,
            }

        self._ready_agents = list(range(len(obs_list)))

        discount_spec = self.discount_spec()
        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self._environment.possible_agents
        }
        observations = self._convert_observations(
            observe, {agent: False for agent in self.possible_agents}
        )
        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self.possible_agents
        }
        discount_spec = self.discount_spec()
        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self.possible_agents
        }

        timestep = parameterized_restart(rewards, self._discounts, observations)

        return timestep, {"s_t": state_list}  # TODO Convert this to correct form

    def step(self, action_dict: Dict) -> Tuple[dm_env.TimeStep, np.array]:
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

        rewards, dones, infos = self._env.step(actions)
        obs_list = self._env.get_obs()

        return_obs = {}
        for i, obs in enumerate(obs_list):
            return_obs[i] = {
                "action_mask": self._env.get_avail_agent_actions(i),
                "obs": obs,
            }
        rewards = {i: rewards / len(obs_list) for i in range(len(obs_list))}
        dones = {i: dones for i in range(len(obs_list))}
        infos = {i: infos for i in range(len(obs_list))}
        self._ready_agents = list(range(len(obs_list)))

        rewards_spec = self.reward_spec()

        #  Handle empty rewards
        if not rewards:
            rewards = {
                agent: convert_np_type(rewards_spec[agent].dtype, 0)
                for agent in self.possible_agents
            }
        else:
            rewards = {
                agent: convert_np_type(rewards_spec[agent].dtype, reward)
                for agent, reward in rewards.items()
            }

        if return_obs:
            return_obs = self._convert_observations(return_obs, dones)

        if self.env_done():
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
        else:
            self._step_type = dm_env.StepType.MID

        timestep = dm_env.TimeStep(
            observation=return_obs,
            reward=rewards,
            discount=self._discounts,
            step_type=self._step_type,
        )

        return timestep  # TODO return global state as well

    def env_done(self) -> bool:
        """
        Returns a bool indicating if all agents in env are done.
        """
        return self._environment.env_done  # TODO Check SMAC has this function

    def observation_spec(self) -> types.Observation:
        observation_specs = {}
        for agent in self._environment.possible_agents:
            observation_specs[agent] = types.OLT(
                observation=_convert_to_spec(
                    self._environment.observation_spaces[agent]
                ),
                legal_actions=_convert_to_spec(self._environment.action_spaces[agent]),
                terminal=specs.Array((1,), np.float32),
            )
        return observation_specs

    def action_spec(self) -> Dict[str, specs.DiscreteArray]:
        action_specs = {}
        for agent in self._environment.possible_agents:
            action_specs[agent] = _convert_to_spec(
                self._environment.action_spaces[agent]
            )
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        reward_specs = {}
        for agent in self._environment.possible_agents:
            reward_specs[agent] = specs.Array((), np.float32)

        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        discount_specs = {}
        for agent in self._environment.possible_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        return {}

    @property
    def agents(self) -> List:
        return self._environment.agents

    @property
    def possible_agents(self) -> List:
        return self._environment.possible_agents

    @property
    def environment(self) -> ParallelEnv:
        """Returns the wrapped environment."""
        return self._environment

    @property
    def current_agent(self) -> Any:
        return self._environment.agent_selection

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment."""
        return getattr(self._environment, name)

    # Note sure we need these next methods. Comes from RLlib wrapper.
    def close(self) -> None:
        """Close the environment"""
        self._env.close()

    def seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
