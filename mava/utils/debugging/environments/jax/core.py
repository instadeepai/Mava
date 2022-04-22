import abc
import copy
import typing
from typing import List, Optional, Union

import chex
import jax
import jax.random as jrand
import jax.numpy as jnp
import numpy as np


@chex.dataclass
class EntityState:
    p_pos: jnp.ndarray = jnp.zeros(2)
    # physical velocity
    p_vel: jnp.ndarray = jnp.zeros(2)


# state of agents (including communication and internal/mental state)
@chex.dataclass
class AgentState(EntityState):
    pass


# action of the agent
@chex.dataclass
class Action:
    # I think the size of this depends on dim_p, but seems to always be 2
    u: jnp.ndarray = jnp.zeros(2)


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
    max_speed: float = jnp.nan
    accel: float = jnp.nan
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
    u_noise: float = jnp.nan
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
            jnp.isnan(agent.u_noise),
            lambda x: jnp.array([0.0, 0.0]),
            lambda agent: jrand.normal(world.key, agent.action.u.shape) * agent.u_noise,
            agent,
        )
        p_force_val = jax.lax.cond(
            agent.movable,
            lambda agent: agent.action.u + noise,
            lambda x: jnp.array([0.0, 0.0]),
            agent,
        )

        p_force.append(p_force_val)
    return p_force


# gather physical forces acting on entities
def apply_environment_force(world: JaxWorld, p_force: List[float]) -> List[float]:
    # simple (but inefficient) collision response

    new_p_force = copy.deepcopy(p_force)
    # TODO WHY DOES world.entities not work here???
    for a, entity_a in enumerate(world.agents):
        for b, entity_b in enumerate(world.agents):
            f_a, f_b = jax.lax.cond(
                b <= a,
                lambda *_: jnp.array([[0.0, 0.0], [0.0, 0.0]]),
                get_collision_force,
                world,
                entity_a,
                entity_b,
            )

            new_p_force[a] = f_a
            new_p_force[b] = f_b
            # if b <= a:
            #     continue

    return new_p_force


def move_entity(world: JaxWorld, entity: Entity, p_force: float) -> Entity:
    def clamp_vel(vel, max_speed):
        speed = jnp.sqrt(jnp.sum(jnp.square(vel)))

        return jax.lax.cond(
            speed > max_speed,
            lambda: (vel / speed) * max_speed,
            lambda: entity.state.p_vel,
        )

    def calc_vel(entity):
        vel = (
            entity.state.p_vel * (1 - world.damping)
            + (p_force / entity.mass) * world.dt
        )

        return jax.lax.cond(
            entity.max_speed is None,
            lambda *_: vel,
            lambda vel, max_speed: clamp_vel(vel, max_speed),
            vel,
            entity.max_speed,
        )

    vel = jax.lax.cond(
        entity.movable & (entity.max_speed is not None),
        calc_vel,
        lambda *_: jnp.zeros(entity.state.p_vel.shape),
        entity,
    )
    pos = entity.state.p_pos + vel * world.dt
    return entity.replace(state=entity.state.replace(p_pos=pos, p_vel=vel))


# integrate physical state
@typing.no_type_check
def integrate_state(world: JaxWorld, p_force: List[float]) -> JaxWorld:
    landmarks = copy.deepcopy(world.landmarks)
    agents = copy.deepcopy(world.agents)

    for i, entity in enumerate(landmarks):
        landmarks[i] = move_entity(world, entity, p_force[i])

    for i, entity in enumerate(agents):
        agents[i] = move_entity(world, entity, p_force[i])

    return world.replace(agents=agents, landmarks=landmarks)


# get collision forces for any contact between two entities
@typing.no_type_check
def get_collision_force(
    self: JaxWorld, entity_a: Entity, entity_b: Entity
) -> List[Union[float, None]]:
    def get_force():
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = jnp.logaddexp(0, -(dist - dist_min) / k) * k
        force = jnp.array(self.contact_force * delta_pos / dist * penetration)

        force_a = jax.lax.cond(entity_a.movable, lambda: force, lambda: jnp.zeros(2))
        force_b = jax.lax.cond(entity_b.movable, lambda: force, lambda: jnp.zeros(2))
        # force_a = +force if entity_a.movable else [jnp.nan, jnp.nan]
        # force_b = -force if entity_b.movable else [jnp.nan, jnp.nan]

        return jnp.array([force_a, force_b])

    return jax.lax.cond(
        # not a collider and don't collide against itself
        jnp.logical_not(entity_a.collide | entity_b.collide) | (entity_a is entity_b),
        lambda: jnp.array([[jnp.nan, jnp.nan], [jnp.nan, jnp.nan]]),
        get_force,
    )
