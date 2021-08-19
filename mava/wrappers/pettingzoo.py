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

"""Wraps a PettingZoo MARL environment to be used as a dm_env environment."""
import copy
from typing import Any, Dict, Iterator, List, Optional, Union

import dm_env
import gym
import numpy as np
from acme import specs
from acme.wrappers.gym_wrapper import _convert_to_spec
from pettingzoo.utils.env import AECEnv, ParallelEnv

try:
    from supersuit import black_death_v1
except ImportError:
    black_death_v1 = None

from mava import types
from mava.utils.sort_utils import sort_str_num
from mava.utils.wrapper_utils import (
    apply_env_wrapper_preprocessers,
    convert_dm_compatible_observations,
    convert_np_type,
    parameterized_restart,
)
from mava.wrappers.env_wrappers import ParallelEnvWrapper, SequentialEnvWrapper


class PettingZooAECEnvWrapper(SequentialEnvWrapper):
    def __init__(
        self,
        environment: AECEnv,
        env_preprocess_wrappers: Optional[List] = [
            # (env_preprocessor, dict_with_preprocessor_params)
            (black_death_v1, None),
        ],
    ):
        """Constructor for sequential PZ wrapper.

        Args:
            environment (AECEnv): sequential PZ env.
            env_preprocess_wrappers (Optional[List], optional): Wrappers
                that preprocess envs. Defaults to [ (black_death_v1, None), ].
        """
        self._environment = environment
        self._reset_next_step = True

        if env_preprocess_wrappers:
            self._environment = apply_env_wrapper_preprocessers(
                self._environment, env_preprocess_wrappers
            )
        self.correct_agent_name()
        self.last_turn_agent = None

    def reset(self) -> dm_env.TimeStep:
        """Resets the env.

        Returns:
            dm_env.TimeStep: dm timestep.
        """
        self._reset_next_step = False
        self._environment.reset()
        self._step_types = {
            agent: dm_env.StepType.FIRST for agent in self.possible_agents
        }
        self._first_step_performed = {agent: False for agent in self.possible_agents}

        observe, _, done, _ = self._environment.last()
        agent = self.current_agent
        observation = self._convert_observation(agent, observe, done)

        self._discount = convert_np_type(self.discount_spec()[agent].dtype, 1)

        reward = convert_np_type(self.reward_spec()[agent].dtype, 0)

        return parameterized_restart(reward, self._discount, observation)

    def step(self, action: Union[int, float]) -> dm_env.TimeStep:
        """Steps in env.

        Args:
            action (Union[int, float]): action for a sequential (single) agent.

        Returns:
            dm_env.TimeStep: dm timestep.
        """
        if self._reset_next_step:
            return self.reset()

        _, _, done, _ = self._environment.last()

        # If current agent is done
        if done:
            self._environment.step(None)
        else:
            self._environment.step(action)

        agent = self.current_agent
        # Reset if all agents are done
        if self.env_done():
            self._reset_next_step = True
            reward = convert_np_type(self.reward_spec()[agent].dtype, 0)
            observation = self._convert_observation(
                agent, self._environment.observe(agent), done
            )
        else:
            #  observation for next agent
            observe, reward, done, info = self._environment.last()

            # Convert rewards to match spec
            reward = convert_np_type(self.reward_spec()[agent].dtype, reward)
            observation = self._convert_observation(agent, observe, done)

        step_type = dm_env.StepType.LAST if done else dm_env.StepType.MID

        return dm_env.TimeStep(
            observation=observation,
            reward=reward,
            discount=self._discount,
            step_type=step_type,
        )

    def env_done(self) -> bool:
        """Check if env is done.

        Returns:
            bool: bool indicating if env is done.
        """
        return not self.agents

    def agent_iter(self, max_iter: int = 2 ** 63) -> Iterator:
        """Agent iterator to loop through agents.

        Args:
            max_iter (int, optional): max iterations. Defaults to 2**63.

        Returns:
            Iterator: agent iter.

        """
        return self._environment.agent_iter(max_iter)

    def _convert_observation(
        self, agent: str, observe: Union[dict, np.ndarray], done: bool
    ) -> types.OLT:
        """Convert PettingZoo observation so it's dm_env compatible.

        Args:
            agent (str): agent str.
            observe (Union[dict, np.ndarray]): agent observation.
            done (bool): is agent done.

        Returns:
            types.OLT: dm olt.
        """

        legals: np.ndarray = None
        observation: np.ndarray = None

        if isinstance(observe, dict) and "action_mask" in observe:
            legals = observe["action_mask"]
            observation = observe["observation"]
        else:
            legals = np.ones(
                self._environment.action_spaces[agent].shape,
                dtype=self._environment.action_spaces[agent].dtype,
            )
            observation = observe
        if observation.dtype == np.int8:
            observation = np.dtype(np.float32).type(
                observation
            )  # observation is not expected to be int8
        if legals.dtype == np.int8:
            legals = np.dtype(np.int64).type(legals)

        observation = types.OLT(
            observation=observation,
            legal_actions=legals,
            terminal=np.asarray([done], dtype=np.float32),
        )
        return observation

    # TODO improve this function.
    def correct_agent_name(self) -> None:
        """Function to correct agent names."""
        self._environment.reset()
        if "tictactoe" in self._environment.metadata["name"]:
            corrected_names = ["player_0", "player_1"]
            self._environment.unwrapped.possible_agents = corrected_names
            self._environment.unwrapped.agents = corrected_names
            self._environment.possible_agents = corrected_names
            self._environment.agents = corrected_names
            previous_names = sort_str_num(list(self.observation_spaces.keys()))

            for corrected_name, prev_name in zip(corrected_names, previous_names):
                self.observation_spaces[corrected_name] = self.observation_spaces[
                    prev_name
                ]
                self.action_spaces[corrected_name] = self.action_spaces[prev_name]
                self.rewards[corrected_name] = self.rewards[prev_name]
                self.dones[corrected_name] = self.dones[prev_name]
                self.infos[corrected_name] = self.infos[prev_name]

                del self.observation_spaces[prev_name]
                del self.action_spaces[prev_name]
                del self.rewards[prev_name]
                del self.dones[prev_name]
                del self.infos[prev_name]

    def observation_spec(self) -> types.Observation:
        """Observations spec.

        Returns:
            types.Observation: spec for observations.
        """
        observation_specs = {}
        for agent in self._environment.possible_agents:
            if isinstance(self._environment.observation_spaces[agent], gym.spaces.Dict):
                obs_space = copy.deepcopy(
                    self._environment.observation_spaces[agent]["observation"]
                )
                legal_actions_space = copy.deepcopy(
                    self._environment.observation_spaces[agent]["action_mask"]
                )
            else:
                obs_space = copy.deepcopy(self._environment.observation_spaces[agent])
                legal_actions_space = copy.deepcopy(
                    self._environment.action_spaces[agent]
                )
            if obs_space.dtype == np.int8:
                obs_space.dtype = np.dtype(np.float32)
            if legal_actions_space.dtype == np.int8:
                legal_actions_space.dtype = np.dtype(np.int64)
            observation_specs[agent] = types.OLT(
                observation=_convert_to_spec(obs_space),
                legal_actions=_convert_to_spec(legal_actions_space),
                terminal=specs.Array((1,), np.float32),
            )
        return observation_specs

    def action_spec(self) -> Dict[str, specs.DiscreteArray]:
        """Action spec.

        Returns:
            Dict[str, specs.DiscreteArray]: spec for actions.
        """
        action_specs = {}
        for agent in self.possible_agents:
            action_specs[agent] = _convert_to_spec(
                self._environment.action_spaces[agent]
            )
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Reward spec.

        Returns:
            Dict[str, specs.Array]: Spec for rewards.
        """
        reward_specs = {}
        for agent in self.possible_agents:
            reward_specs[agent] = specs.Array((), np.float32)

        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Discount spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_specs = {}
        for agent in self.possible_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Extra data spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for extras.
        """
        return {}

    @property
    def agents(self) -> List:
        """Agents still alive in env (not done).

        Returns:
            List: alive agents in env.
        """
        return self._environment.agents

    @property
    def possible_agents(self) -> List:
        """All possible agents in env.

        Returns:
            List: all possible agents in env.
        """
        return self._environment.possible_agents

    @property
    def environment(self) -> AECEnv:
        """Returns the wrapped environment.

        Returns:
            AECEnv: sequential env.
        """
        return self._environment

    @property
    def current_agent(self) -> Any:
        """Current active agent.

        Returns:
            Any: current agent.
        """
        return self._environment.agent_selection

    @property
    def num_agents(self) -> int:
        """Number of agents in env.

        Returns:
            int: number of agents in env.
        """
        return self._environment.num_agents

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)


class PettingZooParallelEnvWrapper(ParallelEnvWrapper):
    """Environment wrapper for PettingZoo MARL environments."""

    def __init__(
        self,
        environment: ParallelEnv,
        env_preprocess_wrappers: Optional[List] = [
            # (env_preprocessor, dict_with_preprocessor_params)
            (black_death_v1, None),
        ],
    ):
        """Constructor for parallel PZ wrapper.

        Args:
            environment (ParallelEnv): parallel PZ env.
            env_preprocess_wrappers (Optional[List], optional): Wrappers
                that preprocess envs. Defaults to [ (black_death_v1, None), ].
        """
        self._environment = environment
        self._reset_next_step = True

        if env_preprocess_wrappers:
            self._environment = apply_env_wrapper_preprocessers(
                self._environment, env_preprocess_wrappers
            )

    def reset(self) -> dm_env.TimeStep:
        """Resets the env.

        Returns:
            dm_env.TimeStep: dm timestep.
        """

        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST
        discount_spec = self.discount_spec()
        observe = self._environment.reset()

        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self.possible_agents
        }

        if type(observe) == tuple:
            observe, env_extras = observe
        else:
            env_extras = {}

        observations = self._convert_observations(
            observe, {agent: False for agent in self.possible_agents}
        )
        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self.possible_agents
        }

        return parameterized_restart(rewards, self._discounts, observations), env_extras

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps in env.

        Args:
            actions (Dict[str, np.ndarray]): actions per agent.

        Returns:
            dm_env.TimeStep: dm timestep
        """

        if self._reset_next_step:
            return self.reset()

        observations, rewards, dones, infos = self._environment.step(actions)

        rewards = self._convert_reward(rewards)
        observations = self._convert_observations(observations, dones)

        if self.env_done():
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True
        else:
            self._step_type = dm_env.StepType.MID

        return dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=self._discounts,
            step_type=self._step_type,
        )

    def env_done(self) -> bool:
        """Check if env is done.

        Returns:
            bool: bool indicating if env is done.
        """
        return not self.agents

    def _convert_reward(self, rewards: Dict[str, float]) -> Dict[str, float]:
        """Convert rewards to be dm_env compatible.

        Args:
            rewards (Dict[str, float]): rewards per agent.
        """
        rewards_spec = self.reward_spec()
        # Handle empty rewards
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
        return rewards

    def _convert_observations(
        self, observes: Dict[str, np.ndarray], dones: Dict[str, bool]
    ) -> types.Observation:
        """Convert PettingZoo observation so it's dm_env compatible.

        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.

        Returns:
            types.Observation: dm compatible observations.
        """
        return convert_dm_compatible_observations(
            observes,
            dones,
            self._environment.action_spaces,
            self._environment.observation_spaces,
            self.env_done(),
            self.possible_agents,
        )

    def observation_spec(self) -> types.Observation:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        observation_specs = {}
        for agent in self.possible_agents:
            observation_specs[agent] = types.OLT(
                observation=_convert_to_spec(
                    self._environment.observation_spaces[agent]
                ),
                legal_actions=_convert_to_spec(self._environment.action_spaces[agent]),
                terminal=specs.Array((1,), np.float32),
            )
        return observation_specs

    def action_spec(self) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Action spec.

        Returns:
            Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]: spec for actions.
        """
        action_specs = {}
        action_spaces = self._environment.action_spaces
        for agent in self.possible_agents:
            action_specs[agent] = _convert_to_spec(action_spaces[agent])
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Reward spec.

        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        reward_specs = {}
        for agent in self.possible_agents:
            reward_specs[agent] = specs.Array((), np.float32)

        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Discount spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_specs = {}
        for agent in self.possible_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Extra data spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for extra data.
        """
        return {}

    @property
    def agents(self) -> List:
        """Agents still alive in env (not done).

        Returns:
            List: alive agents in env.
        """
        return self._environment.agents

    @property
    def possible_agents(self) -> List:
        """All possible agents in env.

        Returns:
            List: all possible agents in env.
        """
        return self._environment.possible_agents

    @property
    def environment(self) -> ParallelEnv:
        """Returns the wrapped environment.

        Returns:
            ParallelEnv: parallel env.
        """
        return self._environment

    @property
    def current_agent(self) -> Any:
        """Current active agent.

        Returns:
            Any: current agent.
        """
        return self._environment.agent_selection

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying environment.

        Args:
            name (str): attribute.

        Returns:
            Any: return attribute from env or underlying env.
        """
        if hasattr(self.__class__, name):
            return self.__getattribute__(name)
        else:
            return getattr(self._environment, name)
