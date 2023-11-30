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

from typing import Callable, Dict, Tuple

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import Observation, State


class MultiAgentWrapper(Wrapper):
    """
    Multi-agent wrapper for environments with configurable reward handling.

    Args:
        env (Environment): The base environment.
        use_individual_reward (bool): If True, the network uses the list of different rewards given.
        aggregate_rewards (bool): If True, aggregates rewards across agents.
        reward_aggregation_function (str): The function for aggregating rewards ("sum" or "mean").
    """

    def __init__(
        self,
        env: Environment,
        use_individual_reward: bool = False,
        aggregate_rewards: bool = False,
        reward_aggregation_function: str = "sum",
    ):
        super().__init__(env)
        self._use_individual_reward = use_individual_reward
        self._aggregate_rewards = aggregate_rewards
        self.reward_aggregation_function = reward_aggregation_function
        self._num_agents = self._env.num_agents
        # Mapping aggregation functions
        self.aggregation_functions: Dict[str, Callable] = {
            "sum": jnp.sum,
            "mean": jnp.mean,
            # Add more functions as needed
        }
        self.aggregate_function: Callable = self.aggregation_functions.get(
            reward_aggregation_function, jnp.sum
        )

    def aggregate_rewards(
        self, timestep: TimeStep, observation: Observation
    ) -> TimeStep[Observation]:
        """
        Aggregate individual rewards across agents using the specified function.
        This method is designed for environments that return a list of individual rewards
        for each agent.
        """
        # Aggregate individual rewards across agents using the specified aggregation function.
        team_reward = self.aggregate_function(timestep.reward)

        # Repeat the aggregated reward for each agent.
        reward = jnp.repeat(team_reward, self._num_agents)
        return timestep.replace(observation=observation, reward=reward)

    def modify_timestep(self, timestep: TimeStep) -> TimeStep[Observation]:
        """Modify the timestep and update the reward based on the specified reward
        handling strategy."""

        # Create a new observation with adjusted step count
        modified_observation = Observation(
            agents_view=timestep.observation.agents_view,
            action_mask=timestep.observation.action_mask,
            step_count=jnp.repeat(timestep.observation.step_count, self._num_agents),
        )
        if self._use_individual_reward:
            # If the environment returns a list of individual rewards and these are used as is:
            return timestep.replace(observation=modified_observation)
        elif self._aggregate_rewards:
            # If the environment returns a list of individual rewards and rewards needs
            # to be aggregated to use a single team_reward:
            return self.aggregate_rewards(timestep, modified_observation)
        else:
            # If the environment returns a single reward, repeat the original reward for each agent
            modified_reward = jnp.repeat(timestep.reward, self._num_agents)
            return timestep.replace(observation=modified_observation, reward=modified_reward)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment. Updates the step count."""
        state, timestep = self._env.reset(key)
        return state, self.modify_timestep(timestep)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment. Updates the step count."""
        state, timestep = self._env.step(state, action)
        return state, self.modify_timestep(timestep)

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the environment."""
        step_count = specs.BoundedArray(
            (self._env.num_agents,),
            jnp.int32,
            [0] * self._env.num_agents,
            [self._env.time_limit] * self._env.num_agents,
            "step_count",
        )
        return self._env.observation_spec().replace(step_count=step_count)
