# Similiar to https://github.com/deepmind/distrax/blob/master/distrax/_src/distributions/epsilon_greedy.py, # noqa: E501
# except masks invalid actions.
"""Epsilon-Greedy distributions with respect to a set of preferences."""

import chex
import jax.numpy as jnp
from distrax._src.distributions import categorical, distribution

Array = chex.Array


def _argmax_with_random_tie_breaking(
    preferences: chex.Array, mask: chex.Array
) -> chex.Array:
    """Compute probabilities greedily with respect to a set of preferences."""
    # Mask invalid prefs
    preferences = jnp.where(
        mask.astype(bool),
        preferences,
        jnp.finfo(preferences.dtype).min,
    )
    optimal_actions = preferences == preferences.max(axis=-1, keepdims=True)
    return optimal_actions / optimal_actions.sum(axis=-1, keepdims=True)


def _mix_probs_with_uniform(
    probs: chex.Array, epsilon: float, mask: chex.Array
) -> chex.Array:
    """Mix an arbitrary categorical distribution with a uniform distribution."""
    num_actions = probs.shape[-1]
    uniform_probs = jnp.ones_like(probs) / num_actions
    # ensure we never chose invalid actions, even when acting randomly.
    uniform_probs = uniform_probs * mask
    return (1 - epsilon) * probs + epsilon * uniform_probs


class EpsilonGreedyWithMask(categorical.Categorical):
    """A Categorical that is ε-greedy with respect to some preferences.

    Given a set of unnormalized preferences, the distribution is a mixture
    of the Greedy and Uniform distribution; with weight (1-ε) and ε, respectively.
    """

    def __init__(
        self,
        preferences: chex.Array,
        epsilon: float,
        mask: chex.Array,
        dtype: jnp.dtype = int,
    ):
        """Initializes an EpsilonGreedy distribution.

        Args:
          preferences: Unnormalized preferences.
          epsilon: Mixing parameter ε.
          mask: Action mask
          dtype: The type of event samples.
        """
        self._preferences = jnp.asarray(preferences)
        self._epsilon = epsilon
        greedy_probs = _argmax_with_random_tie_breaking(self._preferences, mask)
        probs = _mix_probs_with_uniform(greedy_probs, epsilon, mask)
        super().__init__(probs=probs, dtype=dtype)

    @property
    def epsilon(self) -> float:
        """Mixing parameters of the distribution."""
        return self._epsilon

    @property
    def preferences(self) -> chex.Array:
        """Unnormalized preferences."""
        return self._preferences

    def __getitem__(self, index) -> "EpsilonGreedyWithMask":  # type: ignore
        """See `Distribution.__getitem__`."""
        index = distribution.to_batch_shape_index(self.batch_shape, index)
        return EpsilonGreedyWithMask(
            preferences=self.preferences[index], epsilon=self.epsilon, dtype=self.dtype
        )  # type: ignore
