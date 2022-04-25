import copy
from typing import Optional

import typing

import chex
import jax.random

from mava.utils.debugging.environments.jax.core import Agent, Landmark, Entity, EntityId
from mava.utils.debugging.environments.jax.core import JaxWorld
import jax.numpy as jnp
from functools import partial

from mava.utils.debugging.scenario import BaseScenario

RNG = chex.PRNGKey


def make_world(num_agents: int, key: RNG) -> JaxWorld:
    # set any world properties first
    num_landmarks = num_agents
    # add agents
    agents = [
        Agent(name=EntityId(id=i, type=0), collide=True, size=0.15)
        for i in range(num_agents)
    ]
    # add landmarks
    landmarks = [
        Landmark(name=EntityId(id=i, type=1), collide=False, movable=False)
        for i in range(num_landmarks)
    ]

    # make initial conditions
    return JaxWorld(key=key, agents=agents, landmarks=landmarks)


class Scenario(BaseScenario):
    def __init__(self, world: JaxWorld) -> None:
        super().__init__()
        self.world = world
        self.dim_p = world.dim_p

    def make_world(self, num_agents: int) -> JaxWorld:
        # make initial conditions
        return self.reset_world(self.world)

    # @typing.no_type_check
    def reset_world(self, world: JaxWorld) -> JaxWorld:
        def reset_entity(entity: Entity, key: RNG, color):
            pos = jax.random.uniform(key, (self.dim_p,), minval=-1.0, maxval=1.0)
            return entity.replace(
                color=color,
                state=entity.state.replace(p_pos=pos, p_vel=jnp.zeros(self.dim_p)),
            )

        key, *agent_keys = jax.random.split(world.key, len(world.agents) + 1)
        key, *landmark_keys = jax.random.split(key, len(world.landmarks) + 1)

        reset_agent = partial(reset_entity, color=(0.35, 0.35, 0.85))
        reset_landmark = partial(reset_entity, color=(0.25, 0.25, 0.25))

        agents = list(map(reset_agent, world.agents, agent_keys))
        landmarks = list(map(reset_landmark, world.landmarks, landmark_keys))

        # Reset step counter
        return world.replace(
            key=key, agents=agents, landmarks=landmarks, current_step=0
        )

    def is_collision(self, agent1: Agent, agent2: Agent) -> bool:
        delta_pos = agent1.state.p_pos - agent2.state.p_pos
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        dist_min = agent1.size + agent2.size
        return dist < dist_min

    @staticmethod
    def dist(pt1: jnp.ndarray, pt2: jnp.ndarray) -> float:
        return jnp.sqrt(jnp.sum(jnp.square(pt1 - pt2)))

    @typing.no_type_check
    def reward(self, agent: Agent, a_i: int, world: JaxWorld) -> float:
        # Agents are rewarded based on agent distance to its corresponding
        # landmark, penalized for collisions
        rew = 0.0
        # for i in range(len(world.agents)):
        other_agent = world.agents[a_i]
        landmark = world.landmarks[a_i]
        distance = self.dist(other_agent.state.p_pos, landmark.state.p_pos)

        # Cap distance
        distance = jnp.min(jnp.array([distance, 1.0]))
        rew += 1 - distance

        def collision_reward():
            rew = 0
            for other in world.agents:
                if other is agent:
                    continue

                rew += -(self.is_collision(other, agent).astype(int))
                # jax.lax.cond(
                #     self.is_collision(other, agent), lambda: - 1, lambda: 0
                # )
            return rew

        rew += jax.lax.cond(agent.collide, collision_reward, lambda: 0)

        # if agent.collide:
        #     for other in world.agents:
        #         if other is agent:
        #             continue
        #
        #         rew = jax.lax.cond(
        #             self.is_collision(other, agent), lambda: rew - 1, lambda: rew
        #         )

        return rew

    def observation(self, agent: Agent, a_i: int, world: JaxWorld) -> jnp.ndarray:
        # get the position of the agent's target landmark
        target_landmark = world.landmarks[a_i].state.p_pos - agent.state.p_pos

        # Get the other agent and landmark positions
        other_agents_pos = []
        other_landmarks_pos = []
        for i, other_agent in enumerate(world.agents):
            if other_agent is agent:
                continue
            other_agents_pos.append(other_agent.state.p_pos - agent.state.p_pos)
            other_landmarks_pos.append(
                world.landmarks[i].state.p_pos - agent.state.p_pos
            )

        return jnp.concatenate(
            (
                [agent.state.p_vel]
                + [agent.state.p_pos]
                + [jnp.array([world.current_step / 50])]
                + [target_landmark]
                + other_agents_pos
                + other_landmarks_pos
            )  # + comm
        )

    def done(self, agent: Agent, world: JaxWorld) -> bool:
        return world.current_step >= 50

    def seed(self, seed: Optional[int] = None) -> None:
        self.np_rnd.seed(seed)
