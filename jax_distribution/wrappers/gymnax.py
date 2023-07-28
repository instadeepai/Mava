from typing import Tuple

import chex
import jax.numpy as jnp
from flax import struct
from jumanji.environments.routing.robot_warehouse import State
from jumanji.types import TimeStep
from jumanji.wrappers import Wrapper


@struct.dataclass
class LogEnvState:
    env_state: State
    episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_episode_lengths: int


class LogWrapper(Wrapper):
    """Log the episode returns and lengths."""

    def reset(self, key: chex.PRNGKey) -> Tuple[LogEnvState, TimeStep]:
        """Reset the environment."""
        state, timestep = self._env.reset(key)
        state = LogEnvState(state, 0.0, 0, 0.0, 0)
        return state, timestep

    def step(
        self,
        state: State,
        action: jnp.ndarray,
    ) -> Tuple[State, TimeStep]:
        """Step the environment."""
        env_state, timestep = self._env.step(state.env_state, action)

        new_episode_return = state.episode_returns + jnp.mean(timestep.reward)
        new_episode_length = state.episode_lengths + 1
        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - timestep.last()),
            episode_lengths=new_episode_length * (1 - timestep.last()),
            returned_episode_returns=state.returned_episode_returns
            * (1 - timestep.last())
            + new_episode_return * timestep.last(),
            returned_episode_lengths=state.returned_episode_lengths
            * (1 - timestep.last())
            + new_episode_length * timestep.last(),
        )
        return state, timestep
