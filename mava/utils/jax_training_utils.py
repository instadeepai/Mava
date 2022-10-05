import os

import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from chex import Array
from jax.config import config as jax_config


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


def set_growing_gpu_memory_jax() -> None:
    """Solve gpu mem issues.

    More on this - https://jax.readthedocs.io/en/latest/gpu_memory_allocation.html.
    """
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


def set_jax_double_precision() -> None:
    """Set JAX to use double precision.

    This is usually for env that use int64 action space due to the use of spac.Discrete.

    More on this - https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html#double-64bit-precision. # noqa: E501
    """
    jax_config.update("jax_enable_x64", True)
