from typing import Dict, List, Tuple, Union

import dm_env
import jax
import jax.numpy as jnp
import numpy as np
from dm_env import specs
from haiku import one_hot

from mava.types import OLT
from mava.utils.environments.JaxEnvironments.JaxMAWaterworld import MultiAgentWaterWorld
from mava.utils.id_utils import EntityId


class MultiAgentWaterworldWrapper:
    """Environment wrapper for Debugging MARL environments."""

    def __init__(self, environment: MultiAgentWaterWorld):
        self._environment = environment

        self.num_actions = self._environment.act_shape[-1]

        self.num_agents = self._environment.num_agents

        self.action_table = one_hot(jnp.arange(4), 4)

    @property
    def possible_agents(self) -> List:
        return [EntityId(type=0, id=i) for i in range(self.num_agents)]

    @property
    def agents(self):
        return self.possible_agents

    def reset(self, key):
        """Resets the episode."""

        step_type = dm_env.StepType.FIRST

        discounts = {agent: jnp.float32(1.0) for agent in self.possible_agents}
        state = self._environment.reset(key)

        observations = state.obs

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

        # Not sure if the order is always correct otherwise this line isnt needed
        ordered_actions = {agent: actions[str(agent)] for agent in self.possible_agents}

        formatted_actions = jnp.take(
            self.action_table, jnp.array(list(ordered_actions.values())), axis=0
        )

        state, rewards, done = self._environment.step(state, formatted_actions)

        observations = state.obs

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
            done.astype(bool), lambda: dm_env.StepType.LAST, lambda: dm_env.StepType.MID
        )

        discounts = jax.lax.cond(
            done.astype(bool),
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
                observation=jnp.ones(self._environment.obs_shape[1:], jnp.float32),
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
        obs = env_state.obs[EntityId.from_string(agent_info).id]
        return obs

    def render(self, state, mode):
        return self._environment.render(state)
