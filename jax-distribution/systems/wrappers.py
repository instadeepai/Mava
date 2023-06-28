from typing import Optional, Sequence, Tuple, Union

import chex
import jax
import jax.numpy as jnp
import jumanji
from gymnax.environments import environment, spaces
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
from jax import lax
from jumanji.environments.routing.robot_warehouse.types import Observation, State


class MultiDiscrete(spaces.Space):
    def __init__(
        self,
        min_val: Union[int, Sequence[int]],
        max_val: Union[int, Sequence[int]],
        shape: int,
        dtype: jnp.dtype = jnp.int32,
    ) -> None:
        self.min_val = min_val
        self.max_val = max_val + 1
        self.shape = shape
        self.dtype = dtype

        # TODO: Fix this. This is just a basic single agent hack
        self.n = jnp.sum(self.max_val)

    def sample(self, rng: chex.PRNGKey) -> chex.Array:
        """Sample random action uniformly from set of categorical choices."""
        return jax.random.randint(
            rng, shape=self.shape, minval=self.min_val, maxval=self.max_val + 1
        ).astype(self.dtype)

    def contains(self, x: jnp.int_) -> bool:
        """Check whether specific object is within space."""
        # type_cond = isinstance(x, self.dtype)
        # shape_cond = (x.shape == self.shape)
        range_cond = jnp.logical_and(x >= 0, x < self.n)
        return range_cond


class JumanjiToGymnaxWrapper(environment.Environment):
    def __init__(self, jumanji_env):
        super().__init__()
        self.jumanji_env = jumanji_env

    # @property
    # def default_params(self) -> State:
    #     return EnvParams()

    def step_env(
        self, key: chex.PRNGKey, state: State, action: int, params: State
    ) -> Tuple[chex.Array, State, float, bool, dict]:
        del params, key
        state, timestep = self.jumanji_env.step(state, action)
        observation = timestep.observation
        reward = timestep.reward
        done = timestep.last()
        discount = timestep.discount
        return (
            lax.stop_gradient(observation),
            lax.stop_gradient(state),
            reward,
            done,
            {"discount": discount},
        )

    def reset_env(self, key: chex.PRNGKey, params: State) -> Tuple[chex.Array, State]:
        del params
        state, timestep = self.jumanji_env.reset(key)
        return timestep.observation, state

    @property
    def name(self) -> str:
        return repr(self.jumanji_env)

    @property
    def num_actions(self) -> int:
        """Number of actions possible in environment."""
        return self.jumanji_env.action_spec().num_actions

    def action_space(self, params: Optional[State] = None) -> spaces.Discrete:
        """Action space of the environment."""

        jumanji_action_space = self.jumanji_env.action_spec()

        gymnax_action_space = MultiDiscrete(
            min_val=jumanji_action_space.minimum,
            max_val=jumanji_action_space.maximum,
            shape=jumanji_action_space.shape,
            dtype=jnp.int32,
        )

        return gymnax_action_space  # need to convert to gymnax space

    def observation_space(self, params: Optional[State] = None) -> spaces.Box:
        """Observation space of the environment."""
        jumanji_observation_space = self.jumanji_env.observation_spec()

        gymnax_observation_space = spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(
                jumanji_observation_space.agents_view.shape[0]
                * jumanji_observation_space.agents_view.shape[1],
            ),
            dtype=jnp.int32,
        )

        return gymnax_observation_space

    def state_space(self, params: State) -> spaces.Dict:
        """State space of the environment."""
        raise NotImplementedError


env = jumanji.make("RobotWarehouse-v0")

wrapped_env = JumanjiToGymnaxWrapper(env)
x = 0
