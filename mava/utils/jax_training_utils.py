import os
from typing import Any, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from chex import Array
from jax.config import config as jax_config

from mava.types import OLT


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


def compute_running_mean_var_count(stats: Any, batch: jnp.ndarray) -> jnp.ndarray:
    """Updates the running mean, variance and data counts during training.

    stats (Any)   -- dictionary with running mean, var, std, count
    batch (array) -- current batch of data.

    Returns:
        stats (array)
    """

    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]

    mean, var, count = stats["mean"], stats["var"], stats["count"]

    delta = batch_mean - mean
    tot_count = count + batch_count

    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    new_std = jnp.sqrt(new_var)

    return dict(mean=new_mean, var=new_var, std=new_std, count=new_count)


def normalize(stats: Any, batch: jnp.ndarray) -> jnp.ndarray:
    """Normlaise batch of data using the running mean and variance.

    stats (Any)   -- dictionary with running mean, var, std, count.
    batch (array) -- current batch of data.

    Returns:
        denormalize batch (array)
    """

    mean, std = stats["mean"], stats["std"]
    normalize_batch = (batch - mean) / (std + 1e-8)

    return normalize_batch


def denormalize(stats: Any, batch: jnp.ndarray) -> jnp.ndarray:
    """Transform normalized data back into original distribution

    stats (Any)   -- dictionary with running mean, var, count.
    batch (array) -- current batch of data

    Returns:
        denormalize batch (array)
    """

    mean, std = stats["mean"], stats["std"]
    denormalize_batch = batch * std + mean

    return denormalize_batch


def update_and_normalize_observations(stats: Any, observation: OLT) -> Tuple[Any, OLT]:
    """Update running stats and normalise observations

    stats (Dictionary)   -- array with running mean, var, count.
    batch (OLT namespace)   -- current batch of data for a single agent.

    Returns:
        normalize batch (Dictionary)
    """

    obs_shape = observation.observation.shape
    obs = jax.tree_util.tree_map(
        lambda x: jnp.reshape(x, [-1] + list(x.shape[2:])), observation.observation
    )
    upd_stats = compute_running_mean_var_count(stats, obs)
    norm_obs = normalize(upd_stats, obs)

    # reshape before returning
    norm_obs = jnp.reshape(norm_obs, obs_shape)

    return upd_stats, observation._replace(observation=norm_obs)


def normalize_observations(stats: Any, observation: OLT) -> OLT:
    """Normalise a single observation

    stats (Dictionary)   -- array with running mean, var, count.
    batch (OLT namespace) -- current batch of data in for an agent

    Returns:
        denormalize batch (Dictionary)
    """

    # The type casting is required because we need to preserve
    # the data type before the policy info is computed else we will get
    # an error from the table about the type of policy being double instead of float.
    dtype = observation.observation.dtype
    stats_cast = {key: np.array(value, dtype=dtype) for key, value in stats.items()}

    obs = observation.observation
    norm_obs = normalize(stats_cast, obs)

    return observation._replace(observation=norm_obs)


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
