from typing import Tuple
from flax.struct import dataclass
import jax
import jax.numpy as jnp
import chex
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper
from mava.types import Observation, State
import gymnax
from jumanji import specs
from jumanji.types import StepType, TimeStep, restart

@dataclass
class GymState:
    """Wrapper around a JaxMarl state to provide necessary attributes for jumanji environments."""

    state: State
    key: chex.PRNGKey
    t: int

class CartPole(Wrapper):
    def __init__(self):
        self._env, self._env_params = gymnax.make("CartPole-v1")
        self.num_agents = 1
        self._num_agents = 1
        self._timelimit = 500

    def reset(self, key: chex.PRNGKey) -> Tuple[State, TimeStep]:
        """Reset the environment."""
        key, key_step = jax.random.split(key)
        obs, state = self._env.reset(key_step, self._env_params)
        obs = Observation(
            agents_view=jnp.expand_dims(obs, axis=0),
            action_mask=jnp.ones((self._num_agents,2), bool),
            step_count=jnp.zeros((self._num_agents,), dtype=int),
        )

        return GymState(key=key, state=state, t=0), restart(obs, shape=(self._num_agents,))

    def step(self, state: State, action: chex.Array) -> Tuple[State, TimeStep]:
        """Step the environment."""
        key, key_step = jax.random.split(state.key)
        obs, env_state, reward, done, _ = self._env.step(key_step, state.state, action[0], self._env_params)

        state = GymState(key=key, state=env_state, t=state.t+1)

        obs = Observation(
                agents_view=jnp.expand_dims(obs, axis=0),
                action_mask=jnp.ones((self._num_agents,2), bool),
                step_count=jnp.zeros((self._num_agents,), int) + state.t,
        )

        step_type = jax.lax.select(done, StepType.LAST, StepType.MID)
        ts = TimeStep(
            step_type=step_type,
            reward=jnp.expand_dims(reward, axis=0),
            discount=1.0 - jnp.expand_dims(done, axis=0),
            observation=obs,
            # extras=infos,
        )
        return state, ts

    def observation_spec(self) -> specs.Spec[Observation]:
        """Specification of the observation of the environment."""
        agents_view = specs.BoundedArray(
            shape=self._env.observation_space(self._env_params).shape,
            dtype=self._env.observation_space(self._env_params).dtype,
            minimum=self._env.observation_space(self._env_params).low,
            maximum=self._env.observation_space(self._env_params).high,
        )

        action_mask = specs.BoundedArray(
            (self.num_agents, 2), bool, False, True, "action_mask"
        )
        step_count = specs.BoundedArray(
            (self._num_agents,), int, 0, self._timelimit, "step_count"
        )

        spec = specs.Spec(
            Observation,
            "ObservationSpec",
            agents_view=agents_view,
            action_mask=action_mask,
            step_count=step_count,
        )
        return spec
    
    def action_spec(self) -> specs.Spec:
        return specs.MultiDiscreteArray(
                num_values=jnp.full(1, 2), dtype=self._env.action_space(self._env_params).dtype
            )

    def reward_spec(self) -> specs.Array:
        return specs.Array(shape=(self._num_agents,), dtype=float, name="reward")

    def discount_spec(self) -> specs.BoundedArray:
        return specs.BoundedArray(
            shape=(self._num_agents,), dtype=float, minimum=0.0, maximum=1.0, name="discount"
        )