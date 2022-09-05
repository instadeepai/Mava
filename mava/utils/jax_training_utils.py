import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from chex import Array


def action_mask_categorical_policies(
    distribution: tfd.Categorical, mask: Array
) -> tfd.Categorical:
    """TODO Add description"""
    masked_logits = jnp.where(
        mask.astype(bool),
        distribution.logits,
        jnp.finfo(distribution.logits.dtype).min,
    )
    return tfd.Categorical(logits=masked_logits, dtype=distribution.dtype)
