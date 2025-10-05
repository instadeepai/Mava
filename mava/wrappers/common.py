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

from functools import cached_property
from typing import Any, Tuple

import chex
import jax.numpy as jnp
from jumanji import specs
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import MarlEnv, State


class RepeatRewardWrapper(Wrapper, MarlEnv):
    """Wrapper to repeat the reward for each agent."""

    def __init__(self, env: MarlEnv) -> None:
        super().__init__(env)
        assert self._env.reward_spec.shape == (), "Reward spec must be a scalar."
        self.num_agents = env.num_agents

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)
        reward = jnp.repeat(timestep.reward, self.num_agents)
        return state, timestep.replace(reward=reward)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment and repeat the reward."""
        state, timestep = self._env.step(state, action)
        reward = jnp.repeat(timestep.reward, self.num_agents)
        return state, timestep.replace(reward=reward)

    def reward_spec(self) -> specs.Array:
        spec = self._env.reward_spec
        return specs.Array(shape=(self.num_agents,), dtype=spec.dtype, name="reward")


class RepeatDiscountWrapper(Wrapper, MarlEnv):
    """Wrapper to repeat the discount for each agent."""

    def __init__(self, env: MarlEnv) -> None:
        super().__init__(env)
        assert self._env.discount_spec.shape == (), "Discount spec must be a scalar."
        self.num_agents = env.num_agents

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment and repeat the discount."""
        state, timestep = self._env.reset(key)
        discount = jnp.repeat(timestep.discount, self.num_agents)
        return state, timestep.replace(discount=discount)

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment and repeat the discount."""
        state, timestep = self._env.step(state, action)
        discount = jnp.repeat(timestep.discount, self.num_agents)
        return state, timestep.replace(discount=discount)

    def discount_spec(self) -> specs.BoundedArray:
        """The discount spec."""
        spec = self._env.discount_spec
        return specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=spec.dtype,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )


class EmptyMetricsWrapper(Wrapper, MarlEnv):
    """Wrapper to add an empty env_metrics dict to extras."""

    def _add_empty_metrics(self, timestep: TimeStep) -> TimeStep:
        if "env_metrics" not in timestep.extras:
            timestep.extras["env_metrics"] = {}
        return timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment and add empty metrics."""
        state, timestep = self._env.step(state, action)
        return state, self._add_empty_metrics(timestep)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment and add empty metrics."""
        state, timestep = self._env.reset(key)
        return state, self._add_empty_metrics(timestep)


class RepeatStepCountWrapper(Wrapper, MarlEnv):
    """Wrapper to repeat the step_count in the observation for each agent."""

    def __init__(self, env: MarlEnv) -> None:
        super().__init__(env)
        obs_spec = self._env.observation_spec
        assert "step_count" in obs_spec._specs, "Observation spec must have a step_count field."
        assert obs_spec.step_count.shape == (), "Step count spec must be a scalar."
        self.num_agents = env.num_agents
        self.time_limit = env.time_limit

    def _modify_obs(self, observation: Any) -> Any:
        new_step_count = jnp.repeat(observation.step_count, self.num_agents)
        observation = observation._replace(step_count=new_step_count)
        return observation

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment and repeat the step_count."""
        state, timestep = self._env.step(state, action)
        observation = self._modify_obs(timestep.observation)
        return state, timestep.replace(observation=observation)

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment and repeat the step_count."""
        state, timestep = self._env.reset(key)
        observation = self._modify_obs(timestep.observation)
        return state, timestep.replace(observation=observation)

    @cached_property
    def observation_spec(self) -> specs.Spec:
        """The observation spec."""
        obs_spec = self._env.observation_spec
        obs_spec.step_count = specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=obs_spec.specs["step_count"].dtype,
            minimum=0,
            maximum=self.time_limit,
            name="step_count",
        )
        return obs_spec
