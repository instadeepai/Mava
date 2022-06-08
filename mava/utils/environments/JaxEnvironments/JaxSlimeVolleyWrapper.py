from typing import Dict, List, Tuple, Union

import dm_env
import jax
import jax.numpy as jnp
import numpy as np
from dm_env import specs

from mava.types import OLT
from mava.utils.environments.JaxEnvironments.JaxSlimeVolley import SlimeVolley
from mava.utils.id_utils import EntityId


class SlimeVolleyWrapper:
    """Environment wrapper for Debugging MARL environments."""

    def __init__(
        self,
        environment: SlimeVolley,
        is_multi_agent: bool = True,
        is_cooperative: bool = False,
    ):
        self._environment = environment
        self.action_table = jnp.array(
            [
                [0, 0, 0],  # NOOP
                [1, 0, 0],  # LEFT (forward)
                [1, 0, 1],  # UPLEFT (forward jump)
                [0, 0, 1],  # UP (jump)
                [0, 1, 1],  # UPRIGHT (backward jump)
                [0, 1, 0],  # RIGHT (backward)
            ],
            jnp.float32,
        )

        self.num_actions = len(self.action_table)
        self.is_multi_agent = is_multi_agent
        self.is_cooperative = is_cooperative

        self.num_agents = 2 if self.is_multi_agent else 1

    @property
    def possible_agents(self) -> List:
        return [EntityId(type=0, id=i) for i in range(self.num_agents)]

    def reset(self, key):
        """Resets the episode."""

        step_type = dm_env.StepType.FIRST

        discounts = {agent: jnp.float32(1.0) for agent in self.possible_agents}
        state = self._environment.reset(key)
        right_observation = state.obs_right
        left_observation = state.obs_left
        observations = [right_observation, left_observation]

        rewards = {agent: jnp.float32(0.0) for agent in self.possible_agents}

        observations = {
            agent: OLT(
                observations[EntityId.from_string(agent).id],
                jnp.ones(self.num_actions, int),
                jnp.bool_(False),
            )
            for agent in self.possible_agents
        }

        return state, dm_env.TimeStep(step_type, rewards, discounts, observations), {}

    def step(
        self, state, actions: Dict[str, np.ndarray]
    ) -> Union[dm_env.TimeStep, Tuple[dm_env.TimeStep, Dict[str, np.ndarray]]]:
        """Steps the environment."""

        discrete_right_action = actions[self.possible_agents[0]]
        right_action = jnp.take(self.action_table, discrete_right_action, axis=0)

        def left_step():
            discrete_left_action = actions[self.possible_agents[-1]]
            left_action = jnp.take(self.action_table, discrete_left_action, axis=0)

            return self._environment.step(state, right_action, left_action)

        def right_step():
            return self._environment.step(state, right_action)

        state, reward_right, reward_left, done = jax.lax.cond(
            self.is_multi_agent, lambda: left_step(), lambda: right_step()
        )

        reward_right, reward_left = jax.lax.cond(
            self.is_cooperative,
            lambda: (-jnp.abs(reward_right), -jnp.abs(reward_left)),
            lambda: (reward_right, reward_left),
        )

        right_observation = state.obs_right
        left_observation = state.obs_left
        observations = [right_observation, left_observation]
        rewards = [reward_right, reward_left]

        rewards = {
            agent: jnp.float32(rewards[EntityId.from_string(agent).id])
            for agent in self.possible_agents
        }

        observations = {
            agent: OLT(
                observations[EntityId.from_string(agent).id],
                jnp.ones(self.num_actions, int),
                jnp.bool_(done),
            )
            for agent in self.possible_agents
        }

        step_type = jax.lax.cond(
            done, lambda: dm_env.StepType.LAST, lambda: dm_env.StepType.MID
        )

        discounts = jax.lax.cond(
            done,
            lambda: {agent: jnp.float32(0.0) for agent in self.possible_agents},
            lambda: {agent: jnp.float32(1.0) for agent in self.possible_agents},
        )

        timestep = dm_env.TimeStep(
            observation=observations,
            reward=rewards,
            discount=discounts,
            step_type=step_type,
        )

        return state, timestep, {}

    def observation_spec(self) -> Dict[str, OLT]:
        observation_specs = {}
        for agent in self.possible_agents:

            # Legals spec
            legals = jnp.ones(
                self.num_actions,
                dtype=int,
            )

            observation_specs[agent] = OLT(
                observation=jnp.ones(self._environment.obs_shape, jnp.float32),
                legal_actions=legals,
                terminal=jnp.bool_(True),
            )
        return observation_specs

    def action_spec(self):
        action_specs = {}
        for agent in self.possible_agents:
            action_specs[agent] = specs.DiscreteArray(
                num_values=self.num_actions, dtype=jnp.int32
            )
        return action_specs

    def reward_spec(self) -> Dict[str, specs.Array]:
        """Reward spec.

        Returns:
            Dict[str, specs.Array]: spec for rewards.
        """
        reward_specs = {}
        for agent in self.possible_agents:
            reward_specs[agent] = specs.Array((), jnp.float32)

        return reward_specs

    def discount_spec(self) -> Dict[str, specs.BoundedArray]:
        """Discount spec.

        Returns:
            Dict[str, specs.BoundedArray]: spec for discounts.
        """
        discount_specs = {}
        for agent in self.possible_agents:
            discount_specs[agent] = specs.BoundedArray(
                (), jnp.float32, minimum=0, maximum=1.0
            )
        return discount_specs

    def extra_spec(self) -> Dict[str, specs.BoundedArray]:
        return {}

    def get_agent_mask(self, state, agent_info):
        return jnp.zeros(self.num_actions)

    def get_possible_agents(self):
        return self.possible_agents

    def get_observation(self, env_state, agent_info):
        obs = jax.lax.cond(
            EntityId.from_string(agent_info).id == 0,
            lambda: env_state.obs_right,
            lambda: env_state.obs_left,
        )
        return obs

    def render(self, state, mode):
        return self._environment.render(state)
