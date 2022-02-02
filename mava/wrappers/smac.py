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

"""Wraper for SMAC."""
from typing import Any, Dict, List, Optional, Union

import dm_env
import numpy as np
from acme import specs
from smac.env import StarCraft2Env

from mava import types
from mava.utils.wrapper_utils import convert_np_type, parameterized_restart
from mava.wrappers.env_wrappers import ParallelEnvWrapper


class SMACWrapper(ParallelEnvWrapper):
    """Environment wrapper for PettingZoo MARL environments."""

    def __init__(
        self,
        environment: StarCraft2Env,
        return_state_info: bool = True,
    ):
        """Constructor for parallel PZ wrapper.

        Args:
            environment (ParallelEnv): parallel PZ env.
            env_preprocess_wrappers (Optional[List], optional): Wrappers
                that preprocess envs.
                Format (env_preprocessor, dict_with_preprocessor_params).
            return_state_info: return extra state info
        """
        self._environment = environment
        self._return_state_info = return_state_info
        self._agents = [f"agent_{n}" for n in range(self._environment.n_agents)]

        self._reset_next_step = True
        self._done = False

    def reset(self) -> dm_env.TimeStep:
        """Resets the env.

        Returns:
            dm_env.TimeStep: dm timestep.
        """
        # Reset the environment
        self._environment.reset()
        self._done = False

        self._reset_next_step = False
        self._step_type = dm_env.StepType.FIRST

        # Get observation from env
        observation = self.environment.get_obs()
        legal_actions = self._get_legal_actions()
        observations = self._convert_observations(
            observation, legal_actions, self._done
        )

        # Set env discount to 1 for all agents
        discount_spec = self.discount_spec()
        self._discounts = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self._agents
        }

        # Set reward to zero for all agents
        rewards_spec = self.reward_spec()
        rewards = {
            agent: convert_np_type(rewards_spec[agent].dtype, 0)
            for agent in self._agents
        }

        # Possibly add state information to extras
        if self._return_state_info:
            state = self.get_state()
            extras = {"s_t": state}
        else:
            extras = {}

        return parameterized_restart(rewards, self._discounts, observations), extras

    def step(self, actions: Dict[str, np.ndarray]) -> dm_env.TimeStep:
        """Steps in env.

        Args:
            actions (Dict[str, np.ndarray]): actions per agent.

        Returns:
            dm_env.TimeStep: dm timestep
        """
        # Possibly reset the environment
        if self._reset_next_step:
            return self.reset()

        # Convert dict of actions to list for SMAC
        smac_actions = list(actions.values())

        # Step the SMAC environment
        reward, self._done, self._info = self._environment.step(smac_actions)

        # Get the next observations
        next_observations = self._environment.get_obs()
        legal_actions = self._get_legal_actions()
        next_observations = self._convert_observations(
            next_observations, legal_actions, self._done
        )

        # Convert team reward to agent-wise rewards
        rewards = self._convert_reward(reward)

        # Possibly add state information to extras
        if self._return_state_info:
            state = self.get_state()
            extras = {"s_t": state}
        else:
            extras = {}

        if self._done:
            self._step_type = dm_env.StepType.LAST
            self._reset_next_step = True

            # Discount on last timestep set to zero
            self._discounts = {
                agent: convert_np_type(self.discount_spec()[agent].dtype, 0.0)
                for agent in self._agents
            }
        else:
            self._step_type = dm_env.StepType.MID

        # Create timestep object
        timestep = dm_env.TimeStep(
            observation=next_observations,
            reward=rewards,
            discount=self._discounts,
            step_type=self._step_type,
        )

        return timestep, extras

    def env_done(self) -> bool:
        """Check if env is done.

        Returns:
            bool: bool indicating if env is done.
        """
        return self._done

    def _convert_reward(self, reward: float) -> Dict[str, float]:
        """Convert rewards to be dm_env compatible.

        Args:
            reward: rewards per agent.
        """
        rewards_spec = self.reward_spec()
        rewards = {}
        for agent in self._agents:
            rewards[agent] = convert_np_type(rewards_spec[agent].dtype, reward)
        return rewards

    def _get_legal_actions(self) -> List:
        """Get legal actions from the environment."""
        legal_actions = []
        for i, _ in enumerate(self._agents):
            legal_actions.append(
                np.array(self._environment.get_avail_agent_actions(i), dtype="int")
            )
        return legal_actions

    def _convert_observations(
        self, observations: List, legal_actions: List, done: bool
    ) -> types.Observation:
        """Convert SMAC observation so it's dm_env compatible.

        Args:
            observes (Dict[str, np.ndarray]): observations per agent.
            dones (Dict[str, bool]): dones per agent.

        Returns:
            types.Observation: dm compatible observations.
        """
        olt_observations = {}
        for i, agent in enumerate(self._agents):

            olt_observations[agent] = types.OLT(
                observation=observations[i],
                legal_actions=legal_actions[i],
                terminal=np.asarray([done], dtype=np.float32),
            )

        return olt_observations

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        """Function returns extra spec (format) of the env.

        Returns:
            Dict[str, specs.BoundedArray]: extra spec.
        """
        if self._return_state_info:
            return {"s_t": self._environment.get_state()}
        else:
            return {}

    def observation_spec(self) -> Dict[str, types.OLT]:
        """Observation spec.

        Returns:
            types.Observation: spec for environment.
        """
        self._environment.reset()

        observations = self._environment.get_obs()
        legal_actions = self._get_legal_actions()

        observation_specs = {}
        for i, agent in enumerate(self._agents):

            observation_specs[agent] = types.OLT(
                observation=observations[i],
                legal_actions=legal_actions[i],
                terminal=np.asarray([True], dtype=np.float32),
            )

        return observation_specs

    def action_spec(
        self,
    ) -> Dict[str, Union[specs.DiscreteArray, specs.BoundedArray]]:
        """Action spec.

        Returns:
            spec for actions.
        """
        action_specs = {}
        for agent in self._agents:
            action_specs[agent] = specs.DiscreteArray(
                num_values=self._environment.n_actions, dtype=int
            )
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Reward spec.

        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        reward_specs = {}
        for agent in self._agents:
            reward_specs[agent] = specs.Array((), np.float32)
        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Discount spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_specs = {}
        for agent in self._agents:
            discount_specs[agent] = specs.BoundedArray(
                (), np.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def get_stats(self) -> Optional[Dict]:
        """Return extra stats to be logged.

        Returns:
            extra stats to be logged.
        """
        return self._environment.get_stats()

    @property
    def agents(self) -> List:
        """Agents still alive in env (not done).

        Returns:
            List: alive agents in env.
        """
        return self._agents

    @property
    def possible_agents(self) -> List:
        """All possible agents in env.

        Returns:
            List: all possible agents in env.
        """
        return self._agents

    @property
    def environment(self) -> StarCraft2Env:
        """Returns the wrapped environment.

        Returns:
            ParallelEnv: parallel env.
        """
        return self._environment

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
