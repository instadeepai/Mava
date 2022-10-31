from typing import Any, List, Union

import jax.numpy as jnp
import numpy as np
import pytest

from mava.types import OLT
from mava.utils.jax_training_utils import (
    compute_running_mean_var_count,
    construct_norm_axes_list,
    denormalize,
    normalize,
    normalize_observations,
    update_and_normalize_observations,
)


def test_construct_norm_axes_list() -> None:
    """Test if slices element are construced correctly"""

    elements_to_norm: Union[List[Any], None]
    start_axes = 5
    obs_shape = (20,)
    elements_to_norm = [1, 2, [5, 8], (10, 12)]

    return_list = construct_norm_axes_list(start_axes, elements_to_norm, obs_shape)
    expected_list = tuple([slice(6, 7), slice(7, 8), slice(10, 13), slice(15, 17)])
    assert return_list == expected_list

    start_axes = 0
    elements_to_norm = []
    return_list = construct_norm_axes_list(start_axes, elements_to_norm, obs_shape)
    expected_list = tuple([slice(0, 0)])
    assert return_list == expected_list

    elements_to_norm = None
    return_list = construct_norm_axes_list(start_axes, elements_to_norm, obs_shape)
    expected_list = tuple([slice(0, 20)])
    assert return_list == expected_list

    with pytest.raises(ValueError):
        start_axes = 0
        elements_to_norm = [1, 2, [5, 8], (10, 27)]
        return_list = construct_norm_axes_list(start_axes, elements_to_norm, obs_shape)

        elements_to_norm = [1, 2, [5, 8], (27, 30)]
        return_list = construct_norm_axes_list(start_axes, elements_to_norm, obs_shape)


def test_compute_running_mean_var_count() -> None:
    """Test if the running mean, variance and data counts are computed correctly."""

    for (x1, x2, x3) in [
        (np.random.randn(20, 15), np.random.randn(20, 15), np.random.randn(20, 15)),
        (np.random.randn(20, 15), np.random.randn(20, 15), np.random.randn(20, 15)),
    ]:

        stats = dict(
            mean=jnp.zeros(15),
            var=jnp.zeros(15),
            count=jnp.array([1e-4]),
            std=np.ones(15),
        )
        axes = tuple([slice(0, 15)])
        stats = compute_running_mean_var_count(stats, jnp.array(x1), axes=axes)
        stats = compute_running_mean_var_count(stats, jnp.array(x2), axes=axes)
        stats = compute_running_mean_var_count(stats, jnp.array(x3), axes=axes)

        x = jnp.array(np.concatenate([x1, x2, x3], axis=0))
        stats2 = dict(
            mean=jnp.mean(x, axis=0),
            var=jnp.var(x, axis=0),
            count=jnp.array([x.shape[0]]),
        )

        assert jnp.allclose(stats["mean"], stats2["mean"], atol=1e-5)
        assert jnp.allclose(stats["var"], stats2["var"], atol=1e-3)
        assert jnp.allclose(stats["count"], stats2["count"])


def test_compute_running_mean_var_count_axes() -> None:
    """Test if the running stats correctly excludes specified axes"""

    for (x1, x2, x3) in [
        (np.random.randn(20, 15), np.random.randn(20, 15), np.random.randn(20, 15)),
        (np.random.randn(20, 15), np.random.randn(20, 15), np.random.randn(20, 15)),
    ]:

        stats = dict(
            mean=jnp.zeros(15),
            var=jnp.zeros(15),
            count=jnp.array([1e-4]),
            std=np.ones(15),
        )

        normalize_axes = [1, [3, 5], (7, 8)]
        axes = construct_norm_axes_list(0, normalize_axes, (15,))

        stats = compute_running_mean_var_count(stats, jnp.array(x1), axes=axes)
        stats = compute_running_mean_var_count(stats, jnp.array(x2), axes=axes)
        stats = compute_running_mean_var_count(stats, jnp.array(x3), axes=axes)

        indices = np.r_[axes]
        assert not jnp.allclose(stats["mean"][indices], 0, atol=1e-8)
        assert not jnp.allclose(stats["var"][indices], 0, atol=1e-8)
        assert not jnp.allclose(stats["std"][indices], 1, atol=1e-8)

        reverse_axes = [0, 2, 6, (9, 15)]
        r_axes = construct_norm_axes_list(0, reverse_axes, (15,))
        r_indices = np.r_[r_axes]
        assert jnp.allclose(stats["mean"][r_indices], 0, atol=1e-8)
        assert jnp.allclose(stats["var"][r_indices], 0, atol=1e-8)
        assert jnp.allclose(stats["std"][r_indices], 1, atol=1e-8)


def test_normalization() -> None:
    """Test if the normalization does the right thing"""

    x = jnp.array(np.random.randn(15))
    stats = dict(
        mean=jnp.array([0.2]),
        std=jnp.array([2]),
        var=jnp.array([4]),
        count=jnp.array([15]),
    )
    x_norm = normalize(stats, x)
    x_denorm = denormalize(stats, x_norm)

    assert jnp.allclose(x, x_denorm)


def test_update_and_normalize_observations() -> None:
    """Test if the stats are updated correctly"""

    for (x1, x2, x3) in [
        (
            np.random.randn(1, 20, 15),
            np.random.randn(1, 20, 15),
            np.random.randn(1, 20, 15),
        ),
        (
            np.random.randn(1, 20, 15),
            np.random.randn(1, 20, 15),
            np.random.randn(1, 20, 15),
        ),
    ]:
        stats = dict(
            mean=jnp.zeros(15),
            var=jnp.zeros(15),
            count=jnp.array([1e-4]),
            std=np.ones(15),
        )

        axes = tuple([slice(0, 15)])
        obs = OLT(observation=jnp.array(x1), legal_actions=[1], terminal=[0.0])
        stats, _ = update_and_normalize_observations(stats, obs, axes=axes)
        obs = OLT(observation=jnp.array(x2), legal_actions=[1], terminal=[0.0])
        stats, _ = update_and_normalize_observations(stats, obs, axes=axes)
        obs = OLT(observation=jnp.array(x3), legal_actions=[1], terminal=[0.0])
        stats, _ = update_and_normalize_observations(stats, obs, axes=axes)

        x = jnp.array(np.concatenate([x1, x2, x3], axis=0))
        x = jnp.reshape(x, (-1, 15))
        stats2 = dict(
            mean=jnp.mean(x, axis=0),
            var=jnp.var(x, axis=0),
            count=jnp.array([np.prod(x.shape[0])]),
        )

        assert jnp.allclose(stats["mean"], stats2["mean"], atol=1e-5)
        assert jnp.allclose(stats["var"], stats2["var"], atol=1e-3)
        assert jnp.allclose(stats["count"], stats2["count"])


def test_normalize_observations() -> None:
    """Test if normalisation of observations in OLT type works as expected"""

    x = np.random.randn(15)
    obs = OLT(observation=x, legal_actions=[1], terminal=[0.0])
    stats = dict(
        mean=jnp.array([0.2]),
        var=jnp.array([4]),
        count=jnp.array([15]),
        std=jnp.array([2]),
    )
    obs = normalize_observations(stats, obs)
    obs_norm = jnp.array(obs.observation)

    x_norm = normalize(stats, jnp.array(x))

    assert jnp.allclose(x_norm, obs_norm)
