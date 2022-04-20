import copy
from typing import Optional

import typing

import jax.random

from mava.utils.debugging.environments.jax.core import Agent, Landmark
from mava.utils.debugging.environments.jax.core import JaxWorld
import jax.numpy as jnp
import numpy as np

from mava.utils.debugging.scenario import BaseScenario

RNG = jax.random.PRNGKey


class Scenario(BaseScenario):
    def __init__(self) -> None:
        super().__init__()

    def make_world(self, num_agents: int, key: RNG) -> JaxWorld:

        # set any world properties first
        num_landmarks = num_agents
        # add agents
        agents = [
            Agent(name=f"agent {i}", collide=True, size=0.15) for i in range(num_agents)
        ]
        # for i, agent in enumerate(agents):
        #     agent.name = "agent %d" % i
        #     agent.collide = True
        #     agent.size = 0.15
        # add landmarks
        landmarks = [
            Landmark(name=f"landmark {i}", collide=False, moveable=False)
            for i in range(num_landmarks)
        ]

        world = JaxWorld(agents=agents, landmarks=landmarks)

        # make initial conditions
        return self.reset_world(world, key)

    # @typing.no_type_check
    def reset_world(self, world: JaxWorld, key: RNG) -> JaxWorld:
        agents = copy.deepcopy(world.agents)
        landmarks = copy.deepcopy(world.landmarks)

        # random properties for agents
        for i, agent in enumerate(world.agents):
            agents[i].color = jnp.array([0.35, 0.35, 0.85])
            agents[i].state.p_pos = jax.random.uniform(
                key, (world.dim_p,), minval=-1.0, maxval=1.0
            )
            agents[i].state.p_vel = jnp.zeros(world.dim_p)
        # random properties for landmarks
        for i, landmark in enumerate(world.landmarks):
            landmarks[i].color = jnp.array([0.25, 0.25, 0.25])
            landmarks[i].state.p_pos = jax.random.uniform(
                key, (world.dim_p,), minval=-1, maxval=1
            )
            landmarks[i].state.p_vel = jnp.zeros(world.dim_p)

        # Reset step counter
        return world.replace(agents=agents, landmarks=landmarks, current_step=0)

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
        distance = min(distance, 1.0)
        rew += 1 - distance

        if agent.collide:
            for other in world.agents:
                if other is agent:
                    continue
                if self.is_collision(other, agent):
                    rew -= 1

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

        return jnp.array(
            np.concatenate(
                [agent.state.p_vel]
                + [agent.state.p_pos]
                + [[world.current_step / 50]]
                + [target_landmark]
                + other_agents_pos
                + other_landmarks_pos  # + comm
            )
        )

    def done(self, agent: Agent, world: JaxWorld) -> bool:
        return world.current_step >= 50

    def seed(self, seed: Optional[int] = None) -> None:
        self.np_rnd.seed(seed)
