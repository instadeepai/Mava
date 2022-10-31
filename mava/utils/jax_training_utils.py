import os
from typing import Any, Dict, List, Tuple, Union

import jax
import jax.numpy as jnp
import numpy as np
import tensorflow_probability.substrates.jax.distributions as tfd
from chex import Array
from haiku._src.basic import merge_leading_dims
from jax.config import config as jax_config

from mava import constants
from mava.core_jax import SystemExecutor
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


def init_norm_params(stats_shape: Tuple) -> Dict[str, Union[jnp.array, float]]:
    """Initialise normalistion parameters"""

    stats = dict(
        mean=jnp.zeros(shape=stats_shape),
        var=jnp.zeros(shape=stats_shape),
        std=jnp.ones(shape=stats_shape),
        count=jnp.array([1e-4]),
    )

    return stats


def construct_norm_axes_list(
    start_axes: int,
    elements_to_norm: Union[List[Any], None],
    obs_shape: Tuple,
) -> Tuple[slice, ...]:
    """Construt a list of Tuples containing the features on which to apply normalisation

    Args:
        start_axes (int): default axes from which to start,
            this is always 0 unless we used one of the concatenate wrappers.
        elements_to_norm: List of elements to normalize,
            can be a list of ints for specifying each value to normalize
            or can contain a tuple for a range of values to normalize.
        obs_shape (Tuple) --- observations shape

    Returns:
        elements_to_norm: a tuple to be used with np.r_

    The start_axes is 0 unless we use a wrapper like concat_agent_id
    or concat_previous_actions which add one hot encorded vectors
    at the start of the array.
    elements_to_norm corresponds to user specified axes we want to normalise
    We aussume the user does not consider contenation when specifying elements_to_norm.
    If elements_to_norm is None then we nornmalise all the axes
    elements_to_norm can contain single values or lists and tuples which corresponds to
    start and end values of axes slices. eg. [1, 2, [4,7], (9,15)]
    If elements_to_norm is empty we do not normalize anything.
    If start_axes is different from 0 then we need offset all
    the enteries in elements_to_norm by the start_axes
    For the elements_to_norm [1, 2, [4,7], (9,15)] with start_axes = 0
    output is tuple([slice(1,2), slice(2,3), slice(4,7), slice(9,15)]).
    if elements_to_norm = [] and start_axes = 0
    output is tuple([slice(0, 15)]) assuming obs_shape = (15,)
    """

    if elements_to_norm is None:
        return tuple([slice(start_axes, obs_shape[0])])  # selects everything
    elif len(elements_to_norm) == 0:
        return tuple([slice(start_axes, start_axes)])  # selects nothing
    else:
        return_list = []
        starts = []
        ends = []
        for x in elements_to_norm:
            if type(x) == tuple or type(x) == list:
                element = slice(x[0] + start_axes, x[1] + start_axes)
                starts.append(x[0] + start_axes)
                ends.append(x[1] + start_axes)
            else:
                element = slice(x + start_axes, x + start_axes + 1)
                starts.append(x + start_axes)
                ends.append(x + start_axes + 1)
            return_list.append(element)

        if np.max(starts) >= obs_shape[0] or np.max(ends) > obs_shape[0]:
            raise ValueError(
                "Choosen normalization axes will lead to out of bounds index error!"
            )

        return tuple(return_list)


def compute_running_mean_var_count(
    stats: Dict[str, Union[jnp.array, float]],
    batch: jnp.ndarray,
    axes: Any = slice(0, 1),
) -> jnp.ndarray:
    """Updates the running mean, variance and data counts during training.

    Args:
        stats (Any)   -- dictionary with running mean, var, std, count
        batch (array) -- current batch of data.
        axes (tuple of slices) -- which axes to normalise

    Returns:
        stats (array)
    """

    batch_mean = jnp.mean(batch, axis=0)
    batch_var = jnp.var(batch, axis=0)
    batch_count = batch.shape[0]

    mean, var, count = stats["mean"], stats["var"], stats["count"]

    delta = batch_mean - mean
    tot_count = count + batch_count

    mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + jnp.square(delta) * count * batch_count / tot_count
    var = M2 / tot_count
    std = jnp.sqrt(var)
    new_count = tot_count

    # This assumes the all the features we don't want to
    # normalise are all always at the front.
    new_mean = jnp.zeros_like(mean)
    new_var = jnp.zeros_like(var)
    new_std = jnp.ones_like(std)

    indices = np.r_[axes]
    new_mean = new_mean.at[indices].set(mean[indices])
    new_var = new_var.at[indices].set(var[indices])
    new_std = new_std.at[indices].set(std[indices])

    return dict(mean=new_mean, var=new_var, std=new_std, count=new_count)


def normalize(
    stats: Dict[str, Union[jnp.array, float]], batch: jnp.ndarray
) -> jnp.ndarray:
    """Normlaise batch of data using the running mean and variance.

    Args:
        stats (Any)   -- dictionary with running mean, var, std, count.
        batch (array) -- current batch of data.

    Returns:
        denormalize batch (array)
    """

    mean, std = stats["mean"], stats["std"]
    normalize_batch = (batch - mean) / jnp.fmax(std, 1e-6)

    return normalize_batch


def denormalize(
    stats: Dict[str, Union[jnp.array, float]], batch: jnp.ndarray
) -> jnp.ndarray:
    """Transform normalized data back into original distribution

    Args:
        stats (Any)   -- dictionary with running mean, var, count.
        batch (array) -- current batch of data

    Returns:
        denormalize batch (array)
    """

    mean, std = stats["mean"], stats["std"]
    denormalize_batch = batch * jnp.fmax(std, 1e-6) + mean

    return denormalize_batch


def update_and_normalize_observations(
    stats: Dict[str, Union[jnp.array, float]],
    observation: OLT,
    axes: Any = slice(0, 1),
) -> Tuple[Any, OLT]:
    """Update running stats and normalise observations

    Args:
        stats (Dictionary)   -- array with running mean, var, count.
        batch (OLT namespace)   -- current batch of data for a single agent.
        axes (tuple of slices) -- which axes to normalise

    Returns:
        normalize batch (Dictionary)
    """

    obs_shape = observation.observation.shape
    obs = jax.tree_util.tree_map(
        lambda x: merge_leading_dims(x, num_dims=2), observation.observation
    )

    indices = np.r_[axes]
    upd_stats = compute_running_mean_var_count(stats, obs, indices)
    norm_obs = normalize(upd_stats, obs)

    # the following code makes sure we do not normalise
    # death masked observations. This uses the assumption
    # that all death masked agents have zeroed observations
    sum_obs = jnp.sum(obs[:, indices], axis=1)
    mask = jnp.array(sum_obs != 0, dtype=obs.dtype)
    norm_obs = norm_obs.at[:, indices].set(norm_obs[:, indices] * mask[:, None])

    # reshape before returning
    norm_obs = jnp.reshape(norm_obs, obs_shape)

    return upd_stats, observation._replace(observation=norm_obs)


def normalize_observations(
    stats: Dict[str, Union[jnp.array, float]], observation: Any
) -> OLT:
    """Normalise a single observation

    Args:
        stats (Dictionary)   -- array with running mean, var, count.
        observation (OLT namespace) -- current batch of data in for an agent

    Returns:
        denormalize batch (Dictionary)
    """

    # The type casting is required because we need to preserve
    # the data type before the policy info is computed else we will get
    # an error from the table about the type of policy being double instead of float.
    dtype = observation.observation.dtype
    stats_cast = {key: jnp.array(value, dtype=dtype) for key, value in stats.items()}

    obs = observation.observation
    norm_obs = normalize(stats_cast, obs)

    return observation._replace(observation=norm_obs)


def executor_normalize_observation(executor: SystemExecutor, observations: Any) -> Any:
    """Execute the observations normalization before action selection

    Args:
        executor (SystemExecutor) -- an environment executor
        observation (OLT namespace) -- current batch of observations

    Returns:
        observations (OLT namespace)
    """

    observations_stats = executor.store.norm_params[constants.OBS_NORM_STATE_DICT_KEY]
    agents = list(observations.keys())
    death_masked_agents = executor.store.executor_environment.death_masked_agents
    agents_alive = list(set(agents) - set(death_masked_agents))

    for key in agents_alive:
        observations[key] = normalize_observations(
            observations_stats[key], observations[key]
        )

    return observations


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
