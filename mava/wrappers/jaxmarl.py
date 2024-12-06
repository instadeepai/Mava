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
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import cached_property
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import chex
import jax
import jax.numpy as jnp
from brax.envs import State as BraxState
from chex import Array, PRNGKey
from gymnax.environments import spaces as gymnax_spaces
from jaxmarl.environments import SMAX
from jaxmarl.environments import spaces as jaxmarl_spaces
from jaxmarl.environments.mabrax import MABraxEnv
from jaxmarl.environments.mpe.simple_spread import SimpleSpreadMPE
from jaxmarl.environments.multi_agent_env import MultiAgentEnv
from jumanji import specs
from jumanji.types import StepType, TimeStep, restart
from jumanji.wrappers import Wrapper

from mava.types import Observation, ObservationGlobalState, State

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class JaxMarlState:
    """Wrapper around a JaxMarl state to provide necessary attributes for jumanji environments."""

    state: State
    key: chex.PRNGKey
    step: int


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
    spec: Dict[str, Union[jaxmarl_spaces.Box, jaxmarl_spaces.Discrete]],
) -> jaxmarl_spaces.Space:
    """Convert a dictionary of spaces into a single space with a num_agents size first dimension.

    JaxMarl uses a dictionary of specs, one per agent. For now we want this to be a single spec.
    """
    n_agents = len(spec)
    # Get the first agent's spec from the dictionary.
    single_spec = copy.deepcopy(next(iter(spec.values())))

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
        constructor = namedtuple("SubSpace", list(space.spaces.keys()))  # type: ignore
        # Recursively convert spaces to specs
        sub_specs = {
            sub_space_name: jaxmarl_space_to_jumanji_spec(sub_space)
            for sub_space_name, sub_space in space.spaces.items()
        }
        return specs.Spec(constructor=constructor, name="", **sub_specs)
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


class JaxMarlWrapper(Wrapper, ABC):
    """A wrapper for JaxMarl environments to make their API compatible with Jumanji environments."""

    def __init__(
        self,
        env: MultiAgentEnv,
        has_global_state: bool,
        # We set this to -1 to make it an optional input for children of this class.
        # They must set their own defaults or use the wrapped envs value.
        time_limit: int = -1,
    ) -> None:
        """Initialize the JaxMarlWrapper.

        Args:
        ----
        - env: The JaxMarl environment to wrap.
        - has_global_state: Whether the environment has global state.
        - time_limit: The time limit for each episode.
        """
        # Check that all specs are the same as we only support homogeneous environments, for now ;)
        homogenous_error = (
            f"Mava only supports environments with homogeneous agents, "
            f"but you tried to use {env} which is not homogeneous."
        )
        assert is_homogenous(env), homogenous_error
        # Making sure the child envs set this correctly.
        assert time_limit > 0, f"Time limit must be greater than 0, got {time_limit}"

        self.has_global_state = has_global_state
        self.time_limit = time_limit
        super().__init__(env)
        self._env: MultiAgentEnv
        self.agents = self._env.agents
        self.num_agents = self._env.num_agents

        # Calling these on init to cache the values in a non-jitted context.
        self.state_size  # noqa: B018
        self.action_dim  # noqa: B018

    def reset(
        self, key: PRNGKey
    ) -> Tuple[JaxMarlState, TimeStep[Union[Observation, ObservationGlobalState]]]:
        key, reset_key = jax.random.split(key)
        obs, env_state = self._env.reset(reset_key)

        obs = self._create_observation(obs, env_state)
        state = JaxMarlState(env_state, key, jnp.array(0, dtype=int))
        timestep = restart(obs, shape=(self.num_agents,))

        return state, timestep

    def step(
        self, state: JaxMarlState, action: Array
    ) -> Tuple[JaxMarlState, TimeStep[Union[Observation, ObservationGlobalState]]]:
        key, step_key = jax.random.split(state.key)
        obs, env_state, reward, done, _ = self._env.step(
            step_key, state.state, unbatchify(action, self.agents)
        )

        obs = self._create_observation(obs, env_state)
        obs = obs._replace(step_count=jnp.repeat(state.step, self.num_agents))
        step_type = jax.lax.select(done["__all__"], StepType.LAST, StepType.MID)

        ts = TimeStep(
            step_type=step_type,
            reward=batchify(reward, self.agents),
            discount=(1.0 - batchify(done, self.agents)).astype(float),
            observation=obs,
        )
        state = JaxMarlState(env_state, key, state.step + jnp.array(1, dtype=int))

        return state, ts

    def _create_observation(
        self,
        obs: Dict[str, Array],
        wrapped_env_state: Any,
    ) -> Union[Observation, ObservationGlobalState]:
        """Create an observation from the raw observation and environment state."""
        obs_data = {
            "agents_view": batchify(obs, self.agents),
            "action_mask": self.action_mask(wrapped_env_state),
            "step_count": jnp.zeros(self.num_agents, dtype=int),
        }
        if self.has_global_state:
            obs_data["global_state"] = self.get_global_state(wrapped_env_state, obs)
            return ObservationGlobalState(**obs_data)

        return Observation(**obs_data)

    @cached_property
    def observation_spec(self) -> specs.Spec:
        agents_view = jaxmarl_space_to_jumanji_spec(merge_space(self._env.observation_spaces))

        action_mask = specs.BoundedArray(
            (self.num_agents, self.action_dim), bool, False, True, "action_mask"
        )
        step_count = specs.BoundedArray(
            (self.num_agents,), jnp.int32, 0, self.time_limit, "step_count"
        )

        if self.has_global_state:
            global_state = specs.Array(
                (self.num_agents, self.state_size),
                agents_view.dtype,
                "global_state",
            )

            return specs.Spec(
                ObservationGlobalState,
                "ObservationSpec",
                agents_view=agents_view,
                action_mask=action_mask,
                global_state=global_state,
                step_count=step_count,
            )

        return specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )

    @cached_property
    def action_spec(self) -> specs.Spec:
        return jaxmarl_space_to_jumanji_spec(merge_space(self._env.action_spaces))

    @cached_property
    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(self.num_agents,), dtype=float, name="reward")

    @cached_property
    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(self.num_agents,),
            dtype=float,
            minimum=0.0,
            maximum=1.0,
            name="discount",
        )

    @property
    def unwrapped(self) -> MultiAgentEnv:
        return self._env

    @abstractmethod
    def action_mask(self, wrapped_env_state: Any) -> Array:
        """Get action mask for each agent."""
        ...

    @abstractmethod
    def get_global_state(self, wrapped_env_state: Any, obs: Dict[str, Array]) -> Array:
        """Get global state from observation for each agent."""
        ...

    @cached_property
    @abstractmethod
    def action_dim(self) -> chex.Array:
        """Get the actions dim for each agent."""
        ...

    @cached_property
    @abstractmethod
    def state_size(self) -> chex.Array:
        """Get the sate size of the global observation"""
        ...


class SmaxWrapper(JaxMarlWrapper):
    """Wrapper for SMAX environment"""

    def __init__(
        self,
        env: MultiAgentEnv,
        has_global_state: bool = False,
    ):
        super().__init__(env, has_global_state, env.max_steps)
        self._env: SMAX

    def reset(
        self, key: PRNGKey
    ) -> Tuple[JaxMarlState, TimeStep[Union[Observation, ObservationGlobalState]]]:
        state, ts = super().reset(key)
        extras = {"won_episode": False}
        ts = ts.replace(extras=extras)
        return state, ts

    def step(
        self, state: JaxMarlState, action: Array
    ) -> Tuple[JaxMarlState, TimeStep[Union[Observation, ObservationGlobalState]]]:
        state, ts = super().step(state, action)

        current_winner = (ts.step_type == StepType.LAST) & jnp.all(ts.reward >= 1.0)
        extras = {"won_episode": current_winner}
        ts = ts.replace(extras=extras)
        return state, ts

    @cached_property
    def state_size(self) -> chex.Array:
        """Get the sate size of the global observation"""
        return self._env.state_size

    @cached_property
    def action_dim(self) -> chex.Array:
        """Get the actions dim for each agent."""
        single_agent_action_space = self._env.action_space(self.agents[0])
        return single_agent_action_space.n

    def action_mask(self, wrapped_env_state: Any) -> Array:
        """Get action mask for each agent."""
        avail_actions = self._env.get_avail_actions(wrapped_env_state)
        return jnp.array(batchify(avail_actions, self.agents), dtype=bool)

    def get_global_state(self, wrapped_env_state: Any, obs: Dict[str, Array]) -> Array:
        """Get global state from observation and copy it for each agent."""
        return jnp.tile(jnp.array(obs["world_state"]), (self.num_agents, 1))


class MabraxWrapper(JaxMarlWrapper):
    """Wrraper for the Mabrax environment."""

    def __init__(
        self,
        env: MABraxEnv,
        has_global_state: bool = False,
    ):
        super().__init__(env, has_global_state, env.episode_length)
        self._env: MABraxEnv

    @cached_property
    def action_dim(self) -> chex.Array:
        """Get the actions dim for each agent."""
        return self._env.action_space(self.agents[0]).shape[0]

    @cached_property
    def state_size(self) -> chex.Array:
        """Get the sate size of the global observation"""
        brax_env = self._env.env
        return brax_env.observation_size

    def action_mask(self, wrapped_env_state: BraxState) -> Array:
        """Get action mask for each agent."""
        return jnp.ones((self.num_agents, self.action_dim), dtype=bool)

    def get_global_state(self, wrapped_env_state: BraxState, obs: Dict[str, Array]) -> Array:
        """Get global state from observation and copy it for each agent."""
        # Use the global state of brax.
        return jnp.tile(wrapped_env_state.obs, (self.num_agents, 1))


class MPEWrapper(JaxMarlWrapper):
    """Wrapper for the MPE environment."""

    def __init__(
        self,
        env: SimpleSpreadMPE,
        has_global_state: bool = False,
    ):
        super().__init__(env, has_global_state, env.max_steps)
        self._env: SimpleSpreadMPE

    @cached_property
    def action_dim(self) -> chex.Array:
        "Get the actions dim for each agent."
        # Adjusted automatically based on the action_type specified in the kwargs.
        if _is_discrete(self._env.action_space(self.agents[0])):
            return self._env.action_space(self.agents[0]).n
        return self._env.action_space(self.agents[0]).shape[0]

    @cached_property
    def state_size(self) -> chex.Array:
        "Get the state size of the global observation"
        return self._env.observation_space(self.agents[0]).shape[0] * self.num_agents

    def action_mask(self, wrapped_env_state: Any) -> Array:
        """Get action mask for each agent."""
        return jnp.ones((self.num_agents, self.action_dim), dtype=bool)

    def get_global_state(self, wrapped_env_state: Any, obs: Dict[str, Array]) -> Array:
        """Get global state from observation and copy it for each agent."""
        global_state = jnp.concatenate([obs[agent_id] for agent_id in obs])
        return jnp.tile(global_state, (self.num_agents, 1))
