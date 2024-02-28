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

from typing import Tuple

import chex
import gymnax
import jax
import jax.numpy as jnp
from flax.struct import dataclass
from jumanji import specs
from jumanji.types import StepType, TimeStep, restart
from jumanji.wrappers import Wrapper

from mava.types import Observation, State


@dataclass
class GymState:
    """Wrapper around a JaxMarl state to provide necessary attributes for jumanji environments."""

    state: State
    key: chex.PRNGKey
    t: int


class Gymnax(Wrapper):
    def __init__(self, env_identifier):
        self._env, self._env_params = gymnax.make(env_identifier)
        self.num_agents = 1
        self._num_agents = 1
        self._num_actions = self._env.num_actions
        self._timelimit = self._env_params.max_steps_in_episode

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        key, key_step = jax.random.split(key)
        obs, state = self._env.reset(key_step, self._env_params)
        obs = Observation(
            agents_view=jnp.expand_dims(obs, axis=0),
            action_mask=jnp.ones((self._num_agents, self._num_actions), bool),
            step_count=jnp.zeros((self._num_agents,), dtype=int),
        )

        return GymState(key=key, state=state, t=0), restart(obs, shape=(self._num_agents,))

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment."""
        key, key_step = jax.random.split(state.key)
        obs, env_state, reward, done, _ = self._env.step(
            key_step, state.state, action[0], self._env_params
        )

        state = GymState(key=key, state=env_state, t=state.t + 1)

        obs = Observation(
            agents_view=jnp.expand_dims(obs, axis=0),
            action_mask=jnp.ones((self._num_agents, self._num_actions), bool),
            step_count=jnp.zeros((self._num_agents,), int) + state.t,
        )

        step_type = jax.lax.select(done, StepType.LAST, StepType.MID)
        ts = TimeStep(
            step_type=step_type,
            reward=jnp.expand_dims(reward, axis=0),
            discount=1.0 - jnp.expand_dims(done, axis=0),
            observation=obs,
            # extras=infos,
        )
        return state, ts

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the environment."""
        agents_view = specs.BoundedArray(
            shape=self._env.observation_space(self._env_params).shape,
            dtype=self._env.observation_space(self._env_params).dtype,
            minimum=self._env.observation_space(self._env_params).low,
            maximum=self._env.observation_space(self._env_params).high,
        )

        action_mask = specs.BoundedArray(
            (self.num_agents, self._num_actions), bool, False, True, "action_mask"
        )
        step_count = specs.BoundedArray((self._num_agents,), int, 0, self._timelimit, "step_count")

        spec = specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )
        return spec

    def action_spec(self) -> specs.Spec:
        return specs.MultiDiscreteArray(
            num_values=jnp.full(1, self._num_actions),
            dtype=self._env.action_space(self._env_params).dtype,  # oop?
        )

    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(self._num_agents,), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(self._num_agents,), dtype=float, minimum=0.0, maximum=1.0, name="discount"
        )
