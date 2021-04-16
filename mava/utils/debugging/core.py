# Adapted from https://github.com/openai/multiagent-particle-envs.
# TODO (dries): Try using this class directly from PettingZoo and delete this file.

from typing import List, Union

import numpy as np


# physical/external base state of all entites
class EntityState(object):
    def __init__(self) -> None:
        # physical position
        self.p_pos: np.array = None
        # physical velocity
        self.p_vel: np.array = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self) -> None:
        super(AgentState, self).__init__()


# action of the agent
class Action(object):
    def __init__(self) -> None:
        # physical action
        self.u: np.array = None


# properties and state of physical world entity
class Entity(object):
    def __init__(self) -> None:
        # name
        self.name = ""
        # properties:
        self.size = 0.050
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        # material density (affects mass)
        self.density = 25.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 1.0

    @property
    def mass(self) -> float:
        return self.initial_mass


# properties of landmark entities
class Landmark(Entity):
    def __init__(self) -> None:
        super(Landmark, self).__init__()


# properties of agent entities
class Agent(Entity):
    def __init__(self) -> None:
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()


# multi-agent world
class World(object):
    def __init__(self) -> None:
        # list of agents and entities (can change at execution-time!)
        self.agents: List[Agent] = []
        self.landmarks: List[Landmark] = []
        self.current_step = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25
        # contact response parameters
        self.contact_force = 1e2
        self.contact_margin = 1e-3

    # return all entities in the world
    @property
    def entities(self) -> List[Entity]:
        entity_list: List[Entity] = []
        entity_list.extend(self.agents)
        entity_list.extend(self.landmarks)
        return entity_list

    # return all agents controllable by external policies
    @property
    def policy_agents(self) -> List[Agent]:
        return [agent for agent in self.agents]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self) -> List[Agent]:
        return [agent for agent in self.agents]

    # update state of the world
    def step(self) -> None:
        # set actions for scripted agents
        self.current_step += 1
        # apply agent physical controls
        p_force = self.apply_action_force()
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)

    # gather agent action forces
    def apply_action_force(self) -> List[float]:
        # set applied forces
        p_force = []
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                p_force.append(agent.action.u + noise)
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force: List[float]) -> List[float]:
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force: List[float]) -> None:
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel
                        / np.sqrt(
                            np.square(entity.state.p_vel[0])
                            + np.square(entity.state.p_vel[1])
                        )
                        * entity.max_speed
                    )
            entity.state.p_pos += entity.state.p_vel * self.dt

    # get collision forces for any contact between two entities
    def get_collision_force(
        self, entity_a: Entity, entity_b: Entity
    ) -> List[Union[float, None]]:
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        return [force_a, force_b]
