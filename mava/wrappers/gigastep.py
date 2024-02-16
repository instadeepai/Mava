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

from typing import TYPE_CHECKING, Dict, Tuple

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from gigastep.gigastep_env import GigastepEnv
from jumanji import specs
from jumanji.types import StepType, TimeStep, restart
from jumanji.wrappers import Wrapper

from mava.types import Observation, ObservationGlobalState, State

if TYPE_CHECKING:  # https://github.com/python/mypy/issues/6239
    from dataclasses import dataclass
else:
    from flax.struct import dataclass


@dataclass
class GigasteplState:
    """Wrapper around a Gigastep state to provide necessary attributes for jumanji environments."""

    state: State
    key: PRNGKey
    step: int
    Adverseary_action: Array


class GigastepWrapper(Wrapper):
    """
    Wraps a GigaStep environment so that its API is compatible with Jumanji environments.

    Args:
        env: The GigaStep environment to be wrapped.
        time_limit (int): The maximum duration of each episode, in seconds. Defaults to 500.
        has_global_state (bool): Whether the environment has a global state. Defaults to False.
        get_adversary_obs (bool): Whether to retrieve adversary observations for the global state.
        Defaults to False.
        get_adversary_actions (bool): Whether to retrieve adversary actions for the global state.
        Defaults to False.

    Methods:
        reset: Resets the environment and returns a GigasteplState and the restart.
        step: Takes an action and returns the next GigasteplState and timestep.
        action_mask: Returns the action mask for each agent.
        get_global_state: Returns the global state of the environment.
        observation_spec: Returns the observation spec for the environment.
        action_spec: Returns the action spec for the environment.
        reward_spec: Returns the reward spec for the environment.
        discount_spec: Returns the discount spec for the environment.
        _split_obs_and_state: Splits the observation and state into separate arrays for each team.
        get_wining_team: Returns the winning team based on the number of alive agents.
        adversary_policy: Chooses an action for the adversary.
    """

    def __init__(
        self,
        env: GigastepEnv,
        time_limit: int = 500,
        has_global_state: bool = False,
        get_adversary_obs: bool = False,
        get_adversary_actions: bool = False,
    ):

        assert (
            env.discrete_actions
        ), "Only discrete action spaces are currently supported for Gigastep environments"
        assert (
            env._obs_type == "vector"
        ), "Only Vector observations are currently supported for Gigastep environments"
        assert (
            get_adversary_actions is False and get_adversary_obs is False
        ) or has_global_state is True, (
            "For a customized global observation, set has_global_state to True"
        )

        super().__init__(env)
        self._env: GigastepEnv
        self._env.max_episode_length = time_limit
        self.num_agents = self._env.n_agents_team1
        self.num_actions = self._env.n_actions

        self.has_global_state = has_global_state
        self.get_adversary_obs = get_adversary_obs
        self.get_adversary_actions = get_adversary_actions

    def reset(self, key: PRNGKey) -> Tuple[GigasteplState, TimeStep]:
        """
        Reset the GIGASTEP environment.

        Args:
            key (PRNGKey): The PRNGKey.

        Returns:
            GigasteplState : the state of the environment.
            TimeStep : the first time step.

        """

        key, reset_key, adversary_key = jax.random.split(key, 3)
        obs, state = self._env.reset(reset_key)

        team1_obs, team1_state, team2_obs, team2_state = self._split_obs_and_state(obs, state)

        adversary_actions = self.adversary_policy(team2_obs, team2_state, adversary_key)

        if self.has_global_state:
            obs = ObservationGlobalState(
                agents_view=team1_obs,
                action_mask=self.action_mask(),
                global_state=self.get_global_state(obs, adversary_actions),
                step_count=jnp.zeros(self.num_agents, dtype=int),
            )
        else:
            obs = Observation(
                agents_view=team1_obs,
                action_mask=self.action_mask(),
                step_count=jnp.zeros(self.num_agents, dtype=int),
            )

        return GigasteplState(state, key, 0, adversary_actions), restart(
            obs, shape=(self.num_agents,), extras={"won_episode": jnp.nan}
        )

    def step(self, state: GigasteplState, action: Array) -> Tuple[GigasteplState, TimeStep]:
        """
        Takes a step in the Gigastepl environment.

        Args:
            state (GigasteplState): The current state of the environment.
            action (Array): The actions for controllable agents.

        Returns:
            Tuple[GigasteplState, TimeStep]: A tuple containing the next state of the environment
            and the next time step.

        """
        key, step_key, adversary_key = jax.random.split(state.key, 3)

        action = jnp.concatenate([action, state.Adverseary_action], axis=0, dtype=jnp.int16)

        obs, env_state, rewards, dones, ep_done = self._env.step(state.state, action, step_key)

        rewards, dones = (
            rewards[: self.num_agents],
            dones[: self.num_agents],
        )  # cut out the rewards,dones of the adversary
        team1_obs, team1_state, team2_obs, team2_state = self._split_obs_and_state(obs, env_state)
        adversary_actions = self.adversary_policy(
            team2_obs, team2_state, adversary_key
        )  # take the actions of the adversary and cache it befor returning the new state

        if self.has_global_state:
            obs = ObservationGlobalState(
                agents_view=team1_obs,
                action_mask=self.action_mask(),
                global_state=self.get_global_state(obs, adversary_actions),
                step_count=jnp.zeros(self.num_agents, dtype=int),
            )
        else:
            obs = Observation(
                agents_view=team1_obs,
                action_mask=self.action_mask(),
                step_count=jnp.zeros(self.num_agents, dtype=int),
            )

        step_type = jax.lax.select(ep_done, StepType.LAST, StepType.MID)

        current_winner = jnp.where(ep_done, self.get_wining_team(env_state), jnp.nan)

        ts = TimeStep(
            step_type=step_type,
            reward=rewards,
            discount=1.0 - dones,
            observation=obs,
            extras={"won_episode": current_winner},
        )
        return GigasteplState(env_state, key, state.step + 1, adversary_actions), ts

    def action_mask(self) -> Array:
        """Get action mask for each agent."""
        return jnp.ones((self.num_agents, self._env.n_actions))  # all actions are valid

    def get_global_state(self, obs: Array, adversary_actions: Array) -> Array:
        """
        Combines observations from all agents,
        optionally adding adversary actions, and adversary observations
        to create a global state for the environment.

        Args:
            obs (Array): The observations of all agents.
            adversary_actions (Array): The actions for the adversary.

        Returns:
            global_obs (Array): The global observation.
        """
        global_obs = jnp.concatenate(
            obs if self.get_adversary_obs else obs[: self.num_agents], axis=0
        )
        if self.get_adversary_actions:
            global_obs = jnp.concatenate(
                [global_obs, adversary_actions], axis=0
            )  # add the adversery actions to the end of the global observation

        return jnp.tile(global_obs, (self.num_agents, 1))

    def observation_spec(self) -> specs.Spec:
        agents_view = specs.BoundedArray(
            (self.num_agents, *self._env.observation_space.shape),
            jnp.float32,
            -jnp.inf,
            jnp.inf,
            "agents_view",
        )
        action_mask = specs.BoundedArray(
            (self.num_agents, self._env.n_actions), bool, False, True, "action_mask"
        )
        step_count = specs.BoundedArray(
            (self.num_agents,), jnp.int32, 0, self._env.max_episode_length, "step_count"
        )
        if self.has_global_state:
            number_of_agents_in_obs = (
                self._env.n_agents if self.get_adversary_obs else self._env.n_agents_team1
            )  # either the total num of agents or only team1 depending on the flag
            adversary_action_sizes = (
                self._env.n_agents_team2 if self.get_adversary_actions else 0
            )  # consider the added space of the adversary actions
            global_state = specs.BoundedArray(
                (
                    self.num_agents,
                    self._env.observation_space.shape[0] * number_of_agents_in_obs
                    + adversary_action_sizes,
                ),
                jnp.int32,
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

    def action_spec(self) -> specs.Spec:
        return specs.MultiDiscreteArray(num_values=jnp.full(self.num_agents, self.num_actions))

    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(self.num_agents,), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(self.num_agents,), dtype=float, minimum=0.0, maximum=1.0, name="discount"
        )

    def _split_obs_and_state(
        self, obs: Array, state: Tuple[Dict, Dict]
    ) -> Tuple[Array, Tuple[Dict, Dict], Array, Tuple[Dict, Dict]]:
        """
        Separates the observations and state for both teams.

        Args:
            obs (Array): The observations of all agents.
            state (Tuple[Dict, Dict]): The state of all agents.

        Returns:
            Tuple[Array, Tuple[Dict, Dict], Array, Tuple[Dict, Dict]]: Two tuples
            representing observations and states for each team.
        """
        # The first n_agents_team1 elemnts in each array belong to team1
        team1_obs, team2_obs = obs[: self.num_agents], obs[self.num_agents :]

        # split each sub elemnt in the tuple
        team1_state = jax.tree_util.tree_map(lambda x: x[: self.num_agents], state[0])
        team2_state = jax.tree_util.tree_map(lambda x: x[self.num_agents :], state[0])

        return team1_obs, (team1_state, state[1]), team2_obs, (team2_state, state[1])

    def get_wining_team(self, state: Tuple[Dict, Dict]) -> Array:
        """
        Determines the winning team.

        The winning team is the one with more agents alive at the end.

        Args:
            state (Tuple[Dict, Dict]): The state of all agents.

        Returns:
            Array: Winning team indicator (1 if team_1 wins, 0 otherwise).
        """
        # https://github.com/mlech26l/gigastep/blob/main/gigastep/evaluator.py#L261
        alive = state[0]["alive"]
        return jnp.sum(alive[: self.num_agents]) > jnp.sum(alive[self.num_agents :])

    def adversary_policy(self, obs: Array, state: Tuple[Dict, Dict], key: PRNGKey) -> Array:
        """
        Generates actions for the adversary based on observations and state.

        Args:
            obs (Array): The observations of the adversary.
            state (Tuple[Dict, Dict]): The state of the adversary.
            key (PRNGKey): The pseudo-random number generator key.

        Returns:
            Array: Actions for the adversary.
        """
        return jax.random.randint(key, (obs.shape[0],), 0, self.num_actions)
