from typing import Tuple, Union
import chex
import jax
import jax.numpy as jnp

from mava.systems.ppo.types import PPOTransition, RNNPPOTransition


def calculate_gae(
        traj_batch: Union[PPOTransition, RNNPPOTransition], last_val: chex.Array, last_done: chex.Array, gamma: float, gae_lambda: float
) -> Tuple[chex.Array, chex.Array]:
    def _get_advantages(
        carry: Tuple[chex.Array, chex.Array, chex.Array], transition: RNNPPOTransition
    ) -> Tuple[Tuple[chex.Array, chex.Array, chex.Array], chex.Array]:
        gae, next_value, next_done = carry
        done, value, reward = transition.done, transition.value, transition.reward
        gamma = gamma
        delta = reward + gamma * next_value * (1 - next_done) - value
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae
        return (gae, value, done), gae

    _, advantages = jax.lax.scan(
        _get_advantages,
        (jnp.zeros_like(last_val), last_val, last_done),
        traj_batch,
        reverse=True,
        unroll=16,
    )
    return advantages, advantages + traj_batch.value