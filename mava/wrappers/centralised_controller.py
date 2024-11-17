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
import jax.numpy as jnp
from jumanji import specs
from jumanji.env import Environment
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper

from mava.types import (
    Observation,
    State,
)
from mava.utils.centralised_controller import get_all_action_combinations

# NOTE: Rather compute the joint action mask when it is needed instead
# of adding it to the observation


class CentralControllerWrapper(Wrapper):
    def __init__(self, env: Environment):
        super().__init__(env)
        self.num_actions = int(env.action_spec().num_values[0])
        self.joint_action_combinations = get_all_action_combinations(
            self._env.num_agents, self.num_actions
        )

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        state, timestep = self._env.reset(key)

        joint_obs = jnp.concatenate(timestep.observation.agents_view, axis=0)
        timestep.observation = Observation(
            agents_view=joint_obs.astype(float),
            action_mask=timestep.observation.action_mask,
            step_count=timestep.observation.step_count,
        )
        joint_reward = jnp.mean(timestep.reward)
        timestep = timestep.replace(reward=joint_reward)
        return state, timestep

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        team_action = self.joint_action_combinations[action]
        state, timestep = self._env.step(state, team_action)

        joint_obs = jnp.concatenate(timestep.observation.agents_view, axis=0)
        timestep.observation = Observation(
            agents_view=joint_obs.astype(float),
            action_mask=timestep.observation.action_mask,
            step_count=timestep.observation.step_count,
        )
        joint_reward = jnp.mean(timestep.reward)
        timestep = timestep.replace(reward=joint_reward)
        return state, timestep

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the `RobotWarehouse` environment."""
        agents_view = specs.Array(
            (
                self._env.num_agents * self._env.observation_spec().agents_view.shape[1],
            ),  # assume homogeneous agents
            jnp.float32,
            "agents_view",
        )
        action_mask = self._env.observation_spec().action_mask
        step_count = specs.BoundedArray((), jnp.int32, 0, self.time_limit, "step_count")

        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

    def action_spec(self) -> specs.DiscreteArray:
        joint_action_spec = specs.DiscreteArray(
            num_values=self.num_actions**self._env.num_agents,
            name="action",
            dtype=jnp.int32,
        )
        return joint_action_spec
