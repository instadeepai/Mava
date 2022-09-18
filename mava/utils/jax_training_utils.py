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


def compute_running_mean_var_count(stats: Array, batch: Array) -> Array:
    """Updates the running mean, variance and data counts during training.

    stats (array) -- mean, var, count.
    batch (array) -- current batch of data.

    Returns:
        stats (array)
    """

    batch_mean = jnp.mean(batch)
    batch_var = jnp.var(batch)
    batch_count = batch.size

    mean, var, count = stats

    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count

    return jnp.array([new_mean, new_var, new_count])


def normalize(stats: Array, batch: Array) -> Array:
    """Normlaise batch of data using the running mean and variance.

    stats (array) -- mean, var, count.
    batch (array) -- current batch of data.

    Returns:
        denormalize batch (array)
    """

    mean, var, _ = stats
    normalize_batch = (batch - mean) / (jnp.sqrt(jnp.clip(var, a_min=1e-2)))

    return normalize_batch


def denormalize(stats: Array, batch: Array) -> Array:
    """Transform normalized data back into original distribution

    stats (array) -- mean, var, count
    batch (array) -- current batch of data

    Returns:
        denormalize batch (array)
    """

    mean, var, _ = stats
    denormalize_batch = batch * jnp.sqrt(var) + mean

    return denormalize_batch


def set_growing_gpu_memory() -> None:
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
