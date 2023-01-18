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
        death_masking: bool = False,
    ):
        """Constructor for parallel PZ wrapper.

        Args:
            environment (ParallelEnv): parallel PZ env.
            env_preprocess_wrappers (Optional[List], optional): Wrappers
                that preprocess envs.
                Format (env_preprocessor, dict_with_preprocessor_params).
            death_masking: whether to mask out agent observations once dead.
            return_state_info: return extra state info
        """
        self._environment = environment

        self._return_state_info = return_state_info
        self._agents = [f"agent_{n}" for n in range(self._environment.n_agents)]

        # This prevents resetting SMAC if it is already in the reset state. SMAC has a bug
        # where the max returns become less than 20 if reset is called more than once directly
        # after each other.
        self._is_reset = False
        self._done = False

        self._battles_won = 0
        self._battles_game = 0

        self._death_masking = death_masking
        self._pre_agents_alive: Dict[str, Any] = {}

    def reset(self) -> dm_env.TimeStep:
        """Resets the env, if it was not reset on the previous step.

        Returns:
            dm_env.TimeStep: dm timestep.
        """
        # Reset the environment
        if not self._is_reset:
            self._environment.reset()
            self._is_reset = True
        self._done = False

        self._step_type = dm_env.StepType.FIRST

        # Get observation from env
        discount_spec = self.discount_spec()
        agents_mask = {
            agent: convert_np_type(discount_spec[agent].dtype, 1)
            for agent in self._agents
        }
        observation = self.environment.get_obs()
        legal_actions = self._get_legal_actions()
        observations = self._convert_observations(
            observation, legal_actions, agents_mask
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

        self._pre_agents_alive = {agent: True for agent in self._agents}

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
        # Convert dict of actions to list for SMAC
        smac_actions = [actions[agent] for agent in self._agents]

        # Step the SMAC environment
        reward, self._done, self._info = self._environment.step(smac_actions)
        self._is_reset = False

        # Get the next observations
        next_observations = self._environment.get_obs()
        legal_actions = self._get_legal_actions()
        discounts_mask = {}
        for agent in self.possible_agents:
            # If the agent was not done at the start of the episode,
            discounts_mask[agent] = convert_np_type(
                self.discount_spec()[agent].dtype, self._pre_agents_alive[agent]
            )
            self._pre_agents_alive[agent] = not self.is_dead(agent)
        next_observations = self._convert_observations(
            next_observations, legal_actions, discounts_mask
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

    def is_dead(self, agent: Any) -> bool:
        """Check if the agent is dead.

        Returns:
            is_dead: boolean indicating whether the agent is alive or dead.
        """
        return self._environment.agents[int(agent.rsplit("_", -1)[-1])].health == 0.0

    def _convert_observations(
        self, observations: List, legal_actions: List, done
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
            # Check if agent is dead, if so, apply death mask.
            if self._death_masking and self.is_dead(agent):
                observation = np.zeros_like(observations[i])
            else:
                observation = observations[i]

            olt_observations[agent] = types.OLT(
                observation=observation,
                legal_actions=legal_actions[i],
                terminal=np.asarray([done[agent]], dtype=np.float32),
            )

        return olt_observations

    def extras_spec(self) -> Dict[str, specs.BoundedArray]:
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
        if not self._is_reset:
            self._environment.reset()
            self._is_reset = True

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
        stats = self._environment.get_stats()
        stats["cumulative_win_rate"] = stats["win_rate"]
        del stats["win_rate"]
        return stats

    def get_interval_stats(self) -> Optional[Dict]:
        """Computes environment statistics which should be exclusively \
            computed over a given number of evaluation episodes.

        An example would be a win rate calculation where it is required to
        keep track of the number of games won over a given set of evaluation
        episodes.

        Returns:
           Win rate
        """
        interval_stats: Dict[str, Any] = {}
        interval_stats["win_rate"] = (
            self._environment.battles_won - self._battles_won
        ) / (self._environment.battles_game - self._battles_game)
        self._battles_won = self._environment.battles_won
        self._battles_game = self._environment.battles_game
        return interval_stats

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
    def death_masked_agents(self) -> List:
        """Check and returns all death masked agents"""

        masked_agents = []
        for agent in self._agents:
            if self._death_masking and self.is_dead(agent):
                masked_agents.append(agent)

        return masked_agents

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
