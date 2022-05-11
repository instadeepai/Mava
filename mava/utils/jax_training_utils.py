import tensorflow_probability.substrates.jax.distributions as tfd
import jax.numpy as jnp


def action_mask_categorical_policies(distribution, mask):
    masked_logits = jnp.where(
                    mask.astype(bool),
                    distribution.logits,
                    jnp.finfo(distribution.logits.dtype).min,
                )
    return tfd.Categorical(logits=masked_logits)
