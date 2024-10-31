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

from typing import TYPE_CHECKING, Tuple, Union

import chex
import gymnax.environments.spaces as gymnax_spaces
import jax
import jax.numpy as jnp
import numpy as np
from gymnax import EnvParams, EnvState
from gymnax.environments.environment import Environment
from jumanji import specs
from jumanji.specs import Array, DiscreteArray, Spec
from jumanji.types import StepType, TimeStep, restart
from jumanji.wrappers import Wrapper

from mava.types import Observation

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from chex import dataclass


def gymnax_space_to_jumanji_spec(
    space: Union[gymnax_spaces.Discrete, gymnax_spaces.Box, gymnax_spaces.Dict],
) -> Spec:
    """Converts Gymnax spaces to Jumanji specs."""
    if isinstance(space, gymnax_spaces.Discrete):
        return specs.DiscreteArray(num_values=space.n, dtype=int)
    elif isinstance(space, gymnax_spaces.Box):
        # Determine if the space is bounded in all dimensions
        bounded_below = np.all(np.isfinite(space.low))
        bounded_above = np.all(np.isfinite(space.high))
        if bounded_below and bounded_above:
            return specs.BoundedArray(
                shape=space.shape, dtype=space.dtype, minimum=space.low, maximum=space.high
            )
        else:
            # Assume unbounded if any dimension is not bounded
            return specs.Array(shape=space.shape, dtype=space.dtype)
    elif isinstance(space, gymnax_spaces.Dict):
        # Convert nested dict spaces
        dict_specs = {
            key: gymnax_space_to_jumanji_spec(value) for key, value in space.spaces.items()
        }
        return dict_specs
    else:
        raise TypeError(f"Unsupported Gymnax space type: {type(space)}")


@dataclass
class GymnaxEnvState:
    key: chex.PRNGKey
    gymnax_env_state: EnvState
    step_count: chex.Array


class GymnaxWrapper(Wrapper):
    def __init__(self, env: Environment, env_params: EnvParams):
        self._env = env
        self._env_params = env_params
        self.num_agents = 1
        self.time_limit = env_params.max_timesteps
        self.action_dim = env.action_space().n
        if isinstance(self.action_spec(), DiscreteArray):
            n_actions = self.action_spec().num_values
        else:
            n_actions = self.action_spec().shape[0]
        self._legal_action_mask = jnp.ones((n_actions,), dtype=jnp.float32)

    def reset(self, key: chex.PRNGKey) -> Tuple[GymnaxEnvState, TimeStep]:
        key, reset_key = jax.random.split(key)
        obs, gymnax_state = self._env.reset(reset_key, self._env_params)
        obs = Observation(obs, self._legal_action_mask, jnp.array(0))
        timestep = restart(obs, extras={})
        state = GymnaxEnvState(key=key, gymnax_env_state=gymnax_state, step_count=jnp.array(0))
        return state, timestep

    def step(self, state: GymnaxEnvState, action: chex.Array) -> Tuple[GymnaxEnvState, TimeStep]:
        key, key_step = jax.random.split(state.key)
        action = action.squeeze(-1)
        obs, gymnax_state, reward, done, _ = self._env.step(
            key_step, state.gymnax_env_state, action, self._env_params
        )
        state = GymnaxEnvState(
            key=key, gymnax_env_state=gymnax_state, step_count=state.step_count + 1
        )

        timestep = TimeStep(
            observation=Observation(obs, self._legal_action_mask, state.step_count),
            reward=reward,
            discount=jnp.array(1.0 - done),
            step_type=jax.lax.select(done, StepType.LAST, StepType.MID),
            extras={},
        )
        return state, timestep

    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(shape=(), dtype=float, minimum=0.0, maximum=1.0, name="discount")

    def action_spec(self) -> Spec:
        return gymnax_space_to_jumanji_spec(self._env.action_space(self._env_params))

    def observation_spec(self) -> Spec:
        agent_view_spec = gymnax_space_to_jumanji_spec(
            self._env.observation_space(self._env_params)
        )

        action_mask_spec = Array(shape=self._legal_action_mask.shape, dtype=jnp.float32)

        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agent_view_spec,
            action_mask=action_mask_spec,
            step_count=Array(shape=(), dtype=jnp.int32),
        )
