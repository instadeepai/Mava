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

from typing import Dict, List, NamedTuple, Tuple

import jax
import jax.numpy as jnp
from chex import Array, ArrayTree, PRNGKey
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jumanji import specs
from jumanji.types import StepType, TimeStep, restart
from jumanji.wrappers import Wrapper

from mava.types import Observation


def batchify(x: Dict[str, Array], agents: List[str]) -> Array:
    return jnp.stack([x[agent] for agent in agents])


def unbatchify(x: Array, agents: List[str]) -> Dict[str, Array]:
    return {agent: x[i] for i, agent in enumerate(agents)}


class JaxMarlState(NamedTuple):
    state: ArrayTree
    key: PRNGKey
    step: int


class JaxMarlWrapper(Wrapper):
    def __init__(self, env: MultiAgentEnv, timelimit: int = 500):
        super().__init__(env)  # type: ignore
        self._env: MultiAgentEnv
        self.agents = list(self._env.observation_spaces.keys())
        self._timelimit = timelimit
        self._action_shape = self.action_spec().shape

    def reset(self, key: PRNGKey) -> Tuple[JaxMarlState, TimeStep[Observation]]:
        key, reset_key = jax.random.split(key)
        obs, state = self._env.reset(reset_key)

        obs = Observation(
            agents_view=batchify(obs, self.agents),
            action_mask=jnp.ones(self._action_shape),
            step_count=jnp.zeros(self._env.num_agents, dtype=int),
        )
        return JaxMarlState(state, key, 0), restart(obs, extras={}, shape=(self.num_agents,))

    def step(
        self, state: JaxMarlState, action: Array
    ) -> Tuple[JaxMarlState, TimeStep[Observation]]:
        # todo: how do you know if it's a truncation with only done?
        key, step_key = jax.random.split(state.key)
        obs, env_state, reward, done, infos = self._env.step(
            step_key, state.state, unbatchify(action, self.agents)
        )

        step_type = jax.lax.select(done["__all__"], StepType.LAST, StepType.MID)
        ts = TimeStep(
            step_type=step_type,
            reward=batchify(reward, self.agents),
            discount=1.0 - batchify(done, self.agents),
            observation=Observation(
                agents_view=batchify(obs, self.agents),
                action_mask=jnp.ones(self._action_shape),
                step_count=jnp.repeat(state.step, self._env.num_agents),
            ),
            extras=infos,
        )

        return JaxMarlState(env_state, key, state.step + 1), ts

    def observation_spec(self) -> specs.Spec:
        """Returns the observation spec."""
        # todo: this is hard coded for bounded arrays, need to make this general
        single_spec = self._env.observation_space(self.agents[0])
        agents_view = specs.BoundedArray(
            shape=(self._env.num_agents, *single_spec.shape),
            minimum=single_spec.low,
            maximum=single_spec.high,
            dtype=single_spec.dtype,
            name="observation",
        )
        n_actions = self._env.action_space(self.agents[0]).n
        action_mask = specs.BoundedArray(
            (self._env.num_agents, n_actions), bool, False, True, "action_mask"
        )
        # todo: increment step_count
        step_count = specs.BoundedArray(
            (self._env.num_agents,), jnp.int32, 0, self._timelimit, "step_count"
        )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

    def action_spec(self) -> specs.Spec:
        """Returns the action spec."""
        # todo: hard coded for discrete, need to make this general
        single_spec = self._env.action_space(self.agents[0])
        num_values = jnp.repeat(single_spec.n, self._env.num_agents)
        return specs.MultiDiscreteArray(num_values=num_values, name="action")

    def reward_spec(self) -> specs.Array:
        """Returns the reward spec."""
        return specs.Array(shape=(self._env.num_agents,), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        """Returns the discount spec."""
        return specs.BoundedArray(
            shape=(self.num_agents,), dtype=float, minimum=0.0, maximum=1.0, name="discount"
        )
