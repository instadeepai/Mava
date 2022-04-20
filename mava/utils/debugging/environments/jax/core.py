import abc
import copy
import typing
from typing import List, Optional, Union

import chex
import jax
import jax.random
import numpy as np


@chex.dataclass
class EntityState:
    p_pos: Optional[np.ndarray] = None
    # physical velocity
    p_vel: Optional[np.ndarray] = None


# state of agents (including communication and internal/mental state)
@chex.dataclass
class AgentState(EntityState):
    pass


# action of the agent
@chex.dataclass
class Action:
    u: Optional[np.ndarray] = None


# properties and state of physical world entity
@chex.dataclass
class Entity:
    # name
    name: int = -1
    # properties:
    size: float = 0.050
    # entity can move / be pushed
    movable: bool = False
    # entity collides with others
    collide: bool = True
    # material density (affects mass)
    density: float = 25.0
    # color
    color: typing.Tuple[float, float, float] = None
    # max speed and accel
    max_speed: Optional[float] = None
    accel: Optional[float] = None
    # state
    state: EntityState = EntityState()
    # mass
    initial_mass: float = 1.0

    @property
    def mass(self) -> float:
        return self.initial_mass


# properties of landmark entities
@chex.dataclass
class Landmark(Entity):
    pass


# properties of agent entities
@chex.dataclass
class Agent(Entity):
    # agents are movable by default
    movable: bool = True
    # cannot observe the world
    blind: bool = False
    # physical motor noise amount
    u_noise: Union[float, None] = None
    # control range
    u_range: float = 1.0
    # state
    state: AgentState = AgentState()
    # action
    action: Action = Action()


@chex.dataclass(frozen=True)
class JaxWorld:
    key: jax.random.PRNGKey
    agents: List[Agent]
    landmarks: List[Landmark]
    current_step: int = 0
    # position dimensionality
    dim_p: int = 2
    # color dimensionality
    dim_color: int = 3
    # simulation timestep
    dt: float = 0.1
    # physical damping
    damping: float = 0.25
    # contact response parameters
    contact_force: float = 1e2
    contact_margin: float = 1e-3

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


def step(world: JaxWorld) -> JaxWorld:
    # set actions for scripted agents
    world.replace(current_step=world.current_step + 1)
    # apply agent physical controls
    p_force = apply_action_force(world)
    # apply environment forces
    p_force = apply_environment_force(world, p_force)
    # integrate physical state
    return integrate_state(world, p_force)


# gather agent action forces
@typing.no_type_check
def apply_action_force(world: JaxWorld) -> List[float]:
    # set applied forces
    p_force = []

    for i, agent in enumerate(world.agents):

        noise = jax.lax.cond(
            agent.u_noise,
            lambda agent: np.random.randn(*agent.action.u.shape) * agent.u_noise,
            lambda x: 0.0,
            agent,
        )
        p_force_val = jax.lax.cond(
            agent.movable, lambda agent: agent.action.u + noise, lambda x: 0.0, agent
        )

        p_force.append(p_force_val)
    return p_force


# gather physical forces acting on entities
def apply_environment_force(self: JaxWorld, p_force: List[float]) -> List[float]:
    new_p_force = copy.deepcopy(p_force)  # purity
    # simple (but inefficient) collision response
    for a, entity_a in enumerate(self.entities):
        for b, entity_b in enumerate(self.entities):
            if b <= a:
                continue
            [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
            if f_a is not None:
                if new_p_force[a] is None:
                    new_p_force[a] = 0.0
                new_p_force[a] = f_a + new_p_force[a]
            if f_b is not None:
                if new_p_force[b] is None:
                    new_p_force[b] = 0.0
                new_p_force[b] = f_b + new_p_force[b]
    return new_p_force


def move_entity(world: JaxWorld, entity: Entity, p_force: float) -> Entity:
    if not entity.movable:
        return entity

    entity.state.p_vel = entity.state.p_vel * (1 - world.damping)
    if p_force is not None:
        entity.state.p_vel += (p_force / entity.mass) * world.dt
    if entity.max_speed is not None:
        speed = np.sqrt(
            np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
        )
        if speed > entity.max_speed:
            entity.state.p_vel = (
                entity.state.p_vel
                / np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                * entity.max_speed
            )
    entity.state.p_pos += entity.state.p_vel * world.dt

    return entity


# integrate physical state
@typing.no_type_check
def integrate_state(world: JaxWorld, p_force: List[float]) -> JaxWorld:
    landmarks = copy.deepcopy(world.landmarks)
    agents = copy.deepcopy(world.agents)

    for i, entity in enumerate(landmarks):
        landmarks[i] = move_entity(world, entity, p_force[i])

    for i, entity in enumerate(agents):
        agents[i] = move_entity(world, entity, p_force[i])

    return JaxWorld(
        agents=agents,
        landmarks=landmarks,
        current_step=world.current_step,
        dim_p=world.dim_p,
        dim_color=world.dim_color,
        dt=world.dt,
        damping=world.damping,
        contact_force=world.contact_force,
    )


# get collision forces for any contact between two entities
@typing.no_type_check
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
