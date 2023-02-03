from typing import Optional, Sequence, Tuple

import chex
import haiku as hk  # type: ignore
import jax
import jax.numpy as jnp


class QuantileRegressionHead(hk.Module):
    """A Duelling MLP Q-network."""

    def __init__(
        self,
        num_actions: int,
        hidden_sizes: Sequence[int],
        num_atoms: int = 51,
    ):
        super().__init__(name="duelling_q_network")

        self.hidden_sizes = hidden_sizes

        self.num_actions = num_actions
        self._num_quantiles = 51

    def __call__(
        self, inputs: jnp.ndarray
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Forward pass of the duelling network.
        Args:
          inputs: 2-D tensor of shape [batch_size, embedding_size].
        Returns:
          q_values: 2-D tensor of action values of shape [batch_size, num_actions]
        """

        x = hk.nets.MLP(self.hidden_sizes)(inputs)
        x = hk.Linear(self.num_actions * self._num_quantiles)(x)
        q_dist = x.reshape(-1, self.num_actions, self._num_quantiles)
        q_values = jnp.mean(q_dist, axis=-1)

        return q_values, q_dist
