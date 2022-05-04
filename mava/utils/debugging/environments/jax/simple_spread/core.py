import abc
import copy
import typing
from functools import partial
from typing import List, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import jax.random as jrand

from mava.utils.id_utils import EntityId


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
    name: EntityId
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
    u_noise: jnp.ndarray = jnp.array([jnp.nan, jnp.nan])
    # control range
    u_range: float = 1.0
    # state
    state: AgentState = AgentState()
    # action
    action: Action = Action()


@chex.dataclass(frozen=True)
class JaxWorld:
    key: jax.random.PRNGKey
    agents: jnp.ndarray
    landmarks: jnp.ndarray
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
    # @property
    # def entities(self) -> List[Entity]:
    #     entity_list: List[Entity] = []
    #     entity_list.extend(self.agents)
    #     entity_list.extend(self.landmarks)
    #     return entity_list

    # return all agents controllable by external policies
    # @property
    # def policy_agents(self) -> List[Agent]:
    #     return [agent for agent in self.agents]

    # return all agents controlled by world scripts
    # @property
    # def scripted_agents(self) -> List[Agent]:
    #     return [agent for agent in self.agents]


def step(world: JaxWorld) -> JaxWorld:
    # set actions for scripted agents
    world = world.replace(current_step=world.current_step + 1)
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

    key, *agent_keys = jrand.split(world.key, len(world.agents) + 1)
    for i in range(len(world.agents.size)):
        agent = jax.tree_map(lambda x: x[i], world.agents)
        noise = jax.lax.cond(
            jnp.all(jnp.isnan(agent.u_noise)),
            lambda x: jnp.array([0.0, 0.0]),
            lambda agent: jrand.normal(agent_keys[i], agent.action.u.shape)
            * agent.u_noise,
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
    # Checking agents colliding with agents
    for a_i in range(len(world.agents.size)):
        for b_i in range(len(world.agents.size)):
            entity_a = jax.tree_map(lambda x: x[a_i], world.agents)
            entity_b = jax.tree_map(lambda x: x[b_i], world.agents)

            f_a, f_b = jax.lax.cond(
                b_i <= a_i,
                lambda *_: jnp.array([[0.0, 0.0], [0.0, 0.0]]),
                get_collision_force,
                world,
                entity_a,
                entity_b,
            )

            # ideally this should check if entity is movable, but that is not jittable
            if isinstance(entity_a, Agent):
                new_p_force[a_i] += f_a
            if isinstance(entity_b, Agent):
                new_p_force[b_i] += f_b

    # checking entities colliding with entitites
    for a_i in range(len(world.agents.size)):
        for b_i in range(len(world.landmarks.size)):
            entity_a = jax.tree_map(lambda x: x[a_i], world.agents)
            entity_b = jax.tree_map(lambda x: x[b_i], world.landmarks)

            f_a, f_b = get_collision_force(world, entity_a, entity_b)

            # f_a, f_b = jax.lax.cond(
            #     b_i <= a_i,
            #     lambda *_: jnp.array([[0.0, 0.0], [0.0, 0.0]]),
            #     get_collision_force,
            #     world,
            #     entity_a,
            #     entity_b,
            # )

            if isinstance(entity_a, Agent):
                new_p_force[a_i] += f_a

    return new_p_force


def move_entity(world: JaxWorld, entity: Entity, p_force: float) -> Entity:
    """Calculates the velocity of an entity given its force"""

    def clamp_vel(vel, max_speed):
        speed = jnp.sqrt(jnp.sum(jnp.square(vel)))

        return jax.lax.cond(
            speed > max_speed,
            lambda: (vel / speed) * max_speed,
            lambda: entity.state.p_vel,
        )

    def calc_vel(ent):
        vel = ent.state.p_vel * (1 - world.damping) + (p_force / ent.mass) * world.dt

        return jax.lax.cond(
            jnp.isnan(ent.max_speed),
            lambda *_: vel,
            lambda vel, max_speed: clamp_vel(vel, max_speed),
            vel,
            ent.max_speed,
        )

    vel = jax.lax.cond(
        entity.movable,
        calc_vel,
        lambda *_: jnp.zeros(entity.state.p_vel.shape),
        entity,
    )

    pos = entity.state.p_pos + vel * world.dt
    return entity.replace(state=entity.state.replace(p_pos=pos, p_vel=vel))


# integrate physical state
@typing.no_type_check
def integrate_state(world: JaxWorld, p_force: List[float]) -> JaxWorld:
    partial_move_entity = partial(move_entity, world)
    # only calc movement for agent as landmarks aren't movable and don't receive p_forces
    agents = jax.vmap(partial_move_entity)(world.agents, jnp.asarray(p_force))

    return world.replace(agents=agents)


# get collision forces for any contact between two entities
@typing.no_type_check
def get_collision_force(
    world: JaxWorld, entity_a: Entity, entity_b: Entity
) -> List[Union[float, None]]:
    def get_force():
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = jnp.sqrt(jnp.sum(jnp.square(delta_pos)))
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = world.contact_margin
        penetration = jnp.logaddexp(0, -(dist - dist_min) / k) * k
        force = world.contact_force * delta_pos / dist * penetration

        force_a = jax.lax.cond(entity_a.movable, lambda: force, lambda: jnp.zeros(2))
        force_b = jax.lax.cond(entity_b.movable, lambda: force, lambda: jnp.zeros(2))

        return jnp.array([force_a, force_b])

    return jax.lax.cond(
        # not a collider and don't collide against itself
        jnp.logical_not(entity_a.collide | entity_b.collide) | (entity_a is entity_b),
        lambda: jnp.zeros((2, 2)),
        get_force,
    )
