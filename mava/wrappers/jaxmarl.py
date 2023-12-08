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

import copy
from collections import namedtuple
from typing import Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from gymnax.environments import spaces as gymnax_spaces
from jaxmarl.environments import spaces as jaxmarl_spaces
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jumanji import specs
from jumanji.types import StepType, TimeStep, restart
from jumanji.wrappers import Wrapper

from mava.types import JaxMarlState, Observation


def _is_discrete(space: jaxmarl_spaces.Space) -> bool:
    """JaxMarl sometimes uses gymnax and sometimes uses their own specs, so this is needed."""
    return isinstance(space, (gymnax_spaces.Discrete, jaxmarl_spaces.Discrete))


def _is_box(space: jaxmarl_spaces.Space) -> bool:
    """JaxMarl sometimes uses gymnax and sometimes uses their own specs, so this is needed."""
    return isinstance(space, (gymnax_spaces.Box, jaxmarl_spaces.Box))


def _is_dict(space: jaxmarl_spaces.Space) -> bool:
    """JaxMarl sometimes uses gymnax and sometimes uses their own specs, so this is needed."""
    return isinstance(space, (gymnax_spaces.Dict, jaxmarl_spaces.Dict))


def _is_tuple(space: jaxmarl_spaces.Space) -> bool:
    return isinstance(space, (gymnax_spaces.Tuple, jaxmarl_spaces.Tuple))


def batchify(x: Dict[str, Array], agents: List[str]) -> Array:
    """Stack dictionary values into a single array."""
    return jnp.stack([x[agent] for agent in agents])


def unbatchify(x: Array, agents: List[str]) -> Dict[str, Array]:
    """Split array into dictionary entries."""
    return {agent: x[i] for i, agent in enumerate(agents)}


def merge_space(
    spec: Dict[str, Union[jaxmarl_spaces.Box, jaxmarl_spaces.Discrete]]
) -> jaxmarl_spaces.Space:
    """Convert a dictionary of spaces into a single space with a num_agents size first dimension.

    JaxMarl uses a dictionary of specs, one per agent. For now we want this to be a single spec.
    """
    n_agents = len(spec)
    single_spec = copy.deepcopy(list(spec.values())[0])

    err = f"Unsupported space for merging spaces, expected Box or Discrete, got {type(single_spec)}"
    assert _is_discrete(single_spec) or _is_box(single_spec), err

    new_shape = (n_agents, *single_spec.shape)
    single_spec.shape = new_shape

    return single_spec


def is_homogenous(env: MultiAgentEnv) -> bool:
    """Check that all agents in an environment have the same observation and action spaces.

    Note: currently this is done by checking the shape of the observation and action spaces
    as gymnax/jaxmarl environments do not have a custom __eq__ for their specs.
    """
    agents = list(env.observation_spaces.keys())

    main_agent_obs_shape = env.observation_space(agents[0]).shape
    main_agent_act_shape = env.action_space(agents[0]).shape
    # Cannot easily check low, high and n are the same, without being very messy.
    # Unfortunately gymnax/jaxmarl doesn't have a custom __eq__ for their specs.
    same_obs_shape = all(
        env.observation_space(agent).shape == main_agent_obs_shape for agent in agents[1:]
    )
    same_act_shape = all(
        env.action_space(agent).shape == main_agent_act_shape for agent in agents[1:]
    )

    return same_obs_shape and same_act_shape


def jaxmarl_space_to_jumanji_spec(space: jaxmarl_spaces.Space) -> specs.Spec:
    """Convert a jaxmarl space to a jumanji spec."""
    if _is_discrete(space):
        # jaxmarl have multi-discrete, but don't seem to use it.
        if space.shape == ():
            return specs.DiscreteArray(num_values=space.n, dtype=space.dtype)
        else:
            return specs.MultiDiscreteArray(
                num_values=jnp.full(space.shape, space.n), dtype=space.dtype
            )
    elif _is_box(space):
        return specs.BoundedArray(
            shape=space.shape,
            dtype=space.dtype,
            minimum=space.low,
            maximum=space.high,
        )
    elif _is_dict(space):
        # Jumanji needs something to hold the specs
        contructor = namedtuple("SubSpace", list(space.spaces.keys()))  # type: ignore
        # Recursively convert spaces to specs
        sub_specs = {
            sub_space_name: jaxmarl_space_to_jumanji_spec(sub_space)
            for sub_space_name, sub_space in space.spaces.items()
        }
        return specs.Spec(constructor=contructor, name="", **sub_specs)
    elif _is_tuple(space):
        # Jumanji needs something to hold the specs
        field_names = [f"sub_space_{i}" for i in range(len(space.spaces))]
        constructor = namedtuple("SubSpace", field_names)  # type: ignore
        # Recursively convert spaces to specs
        sub_specs = {
            f"sub_space_{i}": jaxmarl_space_to_jumanji_spec(sub_space)
            for i, sub_space in enumerate(space.spaces)
        }
        return specs.Spec(constructor=constructor, name="", **sub_specs)
    else:
        raise ValueError(f"Unsupported JaxMarl space: {space}")


class JaxMarlWrapper(Wrapper):
    """Wraps a JaxMarl environment so that its API is compatible with jumaji environments."""

    def __init__(self, env: MultiAgentEnv, timelimit: int = 500):
        # Check that all specs are the same as we only support homogeneous environments, for now ;)
        homogenous_error = (
            f"Mava only supports environments with homogeneous agents, "
            f"but you tried to use {env} which is not homogeneous."
        )
        assert is_homogenous(env), homogenous_error

        super().__init__(env)
        self._env: MultiAgentEnv
        self._timelimit = timelimit
        self._action_shape = (self.action_spec().shape[0], int(self.action_spec().num_values[0]))

        self.agents = list(self._env.observation_spaces.keys())

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
        # todo: how do you know if it's a truncation with only dones?
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
        agents_view = jaxmarl_space_to_jumanji_spec(merge_space(self._env.observation_spaces))
        single_agent_action_space = self._env.action_space(self.agents[0])
        # we can't mask continuous actions, so just return a shape of 0 for this
        n_actions = (
            single_agent_action_space.n
            if _is_discrete(self._env.action_space(self._env.agents[0]))
            else 0
        )
        action_mask = specs.BoundedArray(
            (self._env.num_agents, n_actions), bool, False, True, "action_mask"
        )
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
        return jaxmarl_space_to_jumanji_spec(merge_space(self._env.action_spaces))

    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(self._env.num_agents,), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(self.num_agents,), dtype=float, minimum=0.0, maximum=1.0, name="discount"
        )
