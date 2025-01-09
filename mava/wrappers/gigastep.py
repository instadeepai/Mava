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
from typing import TYPE_CHECKING, Dict, Tuple, Union

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from gigastep.gigastep_env import GigastepEnv
from jax import tree
from jumanji import specs
from jumanji.types import StepType, TimeStep, restart
from jumanji.wrappers import Wrapper

from mava.types import Observation, ObservationGlobalState, State

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class GigastepState:
    """Wrapper around a Gigastep state to provide necessary attributes for Jumanji environments."""

    state: State
    key: PRNGKey
    step: int
    adversary_action: Array


class GigastepWrapper(Wrapper):
    """Wraps a Gigastep environment so that its API is compatible with Jumanji environments."""

    def __init__(
        self,
        env: GigastepEnv,
        has_global_state: bool = False,
    ):
        """Args:
        ----
            env: The Gigastep environment to be wrapped.
            time_limit (int): The maximum duration of each episode, in seconds. Defaults to 500.
            has_global_state (bool): Whether the environment has a global state. Defaults to False.

        """
        self.has_global_state = has_global_state
        self.time_limit = env.max_episode_length
        self.num_agents = env.n_agents_team1
        self.action_dim = env.n_actions
        super().__init__(env)
        assert (
            env.discrete_actions
        ), "Only discrete action spaces are currently supported for Gigastep environments"
        assert (
            env._obs_type == "vector"
        ), "Only Vector observations are currently supported for Gigastep environments"

        self._env: GigastepEnv

    def reset(self, key: PRNGKey) -> Tuple[GigastepState, TimeStep]:
        """Reset the Gigastep environment.

        Args:
        ----
            key (PRNGKey): The PRNGKey.

        Returns:
        -------
            GigastepState : the state of the environment.
            TimeStep : the first time step.

        """
        key, reset_key, adversary_key = jax.random.split(key, 3)
        obs, state = self._env.reset(reset_key)

        obs_team1, state_team1, obs_team2, state_team2 = self._split_obs_and_state(obs, state)
        # Adversary actions are decided as soon as the observation is available
        # since the old observations aren't available in the step function
        adversary_actions = self.adversary_policy(obs_team2, state_team2, adversary_key)

        state = GigastepState(state, key, 0, adversary_actions)

        obs = self._create_observation(obs_team1, obs, state)

        timestep = restart(obs, shape=(self.num_agents,), extras={"won_episode": False})
        return state, timestep

    def step(self, state: GigastepState, action: Array) -> Tuple[GigastepState, TimeStep]:
        """Takes a step in the Gigastep environment.

        Args:
        ----
            state (GigastepState): The current state of the environment.
            action (Array): The actions for controllable agents.

        Returns:
        -------
            Tuple[GigastepState, TimeStep]: A tuple containing the next state of the environment
            and the next time step.

        """
        key, step_key, adversary_key = jax.random.split(state.key, 3)

        action = jnp.concatenate([action, state.adversary_action], axis=0, dtype=jnp.int16)

        obs, env_state, rewards, dones, ep_done = self._env.step(state.state, action, step_key)

        # cut out the rewards,dones of the adversary
        rewards, dones = (
            rewards[: self.num_agents],
            dones[: self.num_agents],
        )

        obs_team1, state_team1, obs_team2, state_team2 = self._split_obs_and_state(obs, env_state)

        # take the actions of the adversary and cache it before returning the new state
        adversary_actions = self.adversary_policy(obs_team2, state_team2, adversary_key)

        obs = self._create_observation(obs_team1, obs, state)

        step_type = jax.lax.select(ep_done, StepType.LAST, StepType.MID)

        current_winner = ep_done & self.won_episode(env_state)

        ts = TimeStep(
            step_type=step_type,
            reward=rewards,
            discount=1.0 - dones,
            observation=obs,
            extras={"won_episode": current_winner},
        )
        return GigastepState(env_state, key, state.step + 1, adversary_actions), ts

    def _create_observation(
        self,
        obs: Array,
        obs_full: Array,
        state: GigastepState,
    ) -> Union[Observation, ObservationGlobalState]:
        """Create an observation from the raw observation and environment state."""
        obs_data = {
            "agents_view": obs,
            "action_mask": self.action_mask(),
            "step_count": jnp.repeat(state.step, self.num_agents),
        }

        if self.has_global_state:
            obs_data["global_state"] = self.get_global_state(obs_full)
            return ObservationGlobalState(**obs_data)
        else:
            return Observation(**obs_data)

    def action_mask(self) -> Array:
        """Get action mask for each agent."""
        return jnp.ones((self.num_agents, self._env.n_actions))  # all actions are valid

    def get_global_state(self, obs: Array) -> Array:
        """Combines observations from all agents and adversaries
        to create a global state for the environment.

        Args:
        ----
            obs (Array): The observations of all agents and adversaries.

        Returns:
        -------
            global_obs (Array): The global observation.

        """
        # the global observation needs to be tested once we have better heuristics for adversaries.
        global_obs = jnp.concatenate(obs, axis=0)
        return jnp.tile(global_obs, (self.num_agents, 1))

    @cached_property
    def observation_spec(self) -> specs.Spec:
        agents_view = specs.BoundedArray(
            (self.num_agents, *self._env.observation_space.shape),
            jnp.float32,
            -jnp.inf,
            jnp.inf,
            "agents_view",
        )
        action_mask = specs.BoundedArray(
            (self.num_agents, self._env.n_actions), float, 0.0, 1.0, "action_mask"
        )
        step_count = specs.BoundedArray(
            (self.num_agents,), jnp.int32, 0, self._env.max_episode_length, "step_count"
        )
        if self.has_global_state:
            global_state = specs.BoundedArray(
                (self.num_agents, self._env.observation_space.shape[0] * self._env.n_agents),
                jnp.float32,
                0,
                255,
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
        return specs.MultiDiscreteArray(num_values=jnp.full(self.num_agents, self.action_dim))

    @cached_property
    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(self.num_agents,), dtype=float, name="reward")

    @cached_property
    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(self.num_agents,), dtype=float, minimum=0.0, maximum=1.0, name="discount"
        )

    def _split_obs_and_state(
        self, obs: Array, state: Tuple[Dict, Dict]
    ) -> Tuple[Array, Tuple[Dict, Dict], Array, Tuple[Dict, Dict]]:
        """Separates the observations and state for both teams.

        Args:
        ----
            obs (Array): The observations of all agents.
            state (Tuple[Dict, Dict]): The state of all agents.

        Returns:
        -------
            Tuple[Array, Tuple[Dict, Dict], Array, Tuple[Dict, Dict]]: Two tuples
            representing observations and states for each team.

        """
        # The first n_agents_team1 elements in each array belong to team1
        team1_obs, team2_obs = obs[: self.num_agents], obs[self.num_agents :]

        # split each sub elemnt in the tuple
        per_agent_info, general_state_info = state
        team1_state = tree.map(lambda x: x[: self.num_agents], per_agent_info)
        team2_state = tree.map(lambda x: x[self.num_agents :], per_agent_info)

        return (
            team1_obs,
            (team1_state, general_state_info),
            team2_obs,
            (team2_state, general_state_info),
        )

    def won_episode(self, state: Tuple[Dict, Dict]) -> Array:
        """Determines the winning team.

        The winning team is the one with more agents alive at the end.

        Args:
        ----
            state (Tuple[Dict, Dict]): The state of all agents.

        Returns:
        -------
            Array: Winning team indicator (1 if team_1 wins, 0 otherwise).

        """
        # https://github.com/mlech26l/gigastep/blob/main/gigastep/evaluator.py#L261
        alive = state[0]["alive"]
        return jnp.sum(alive[: self.num_agents]) > jnp.sum(alive[self.num_agents :])

    def adversary_policy(self, obs: Array, state: Tuple[Dict, Dict], key: PRNGKey) -> Array:
        """Generates actions for the adversary based on observations and state.

        Args:
        ----
            obs (Array): The observations of the adversary.
            state (Tuple[Dict, Dict]): The state of the adversary.
            key (PRNGKey): The pseudo-random number generator key.

        Returns:
        -------
            Array: Actions for the adversary.

        """
        return jax.random.randint(key, (obs.shape[0],), 0, self.action_dim)

    @property
    def unwrapped(self) -> GigastepEnv:
        return self._env
