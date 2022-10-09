import jax.numpy as jnp
import numpy as np

from mava.types import OLT
from mava.utils.jax_training_utils import (
    compute_running_mean_var_count,
    denormalize,
    normalize,
    normalize_observations,
    update_and_normalize_observations,
)


def test_compute_running_mean_var_count() -> None:
    """Test if the running mean, variance and data counts are computed correctly."""

    for (x1, x2, x3) in [
        (np.random.randn(20, 15), np.random.randn(20, 15), np.random.randn(20, 15)),
        (np.random.randn(20, 15), np.random.randn(20, 15), np.random.randn(20, 15)),
    ]:

        stats = dict(mean=jnp.zeros(15), var=jnp.zeros(15), count=jnp.array([1e-4]))

        stats = compute_running_mean_var_count(stats, jnp.array(x1))
        stats = compute_running_mean_var_count(stats, jnp.array(x2))
        stats = compute_running_mean_var_count(stats, jnp.array(x3))

        x = jnp.array(np.concatenate([x1, x2, x3], axis=0))
        stats2 = dict(
            mean=jnp.mean(x, axis=0),
            var=jnp.var(x, axis=0),
            count=jnp.array([x.shape[0]]),
        )

        assert jnp.allclose(stats["mean"], stats2["mean"], atol=1e-5)
        assert jnp.allclose(stats["var"], stats2["var"], atol=1e-3)
        assert jnp.allclose(stats["count"], stats2["count"])


def test_normalization() -> None:
    """Test if the normalization does the right thing"""

    x = jnp.array(np.random.randn(15))
    stats = dict(mean=jnp.array([0.2]), var=jnp.array([1.5]), count=jnp.array([15]))
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
        stats = dict(mean=jnp.zeros(15), var=jnp.zeros(15), count=jnp.array([1e-4]))

        obs = OLT(observation=jnp.array(x1), legal_actions=[1], terminal=[0.0])
        stats, _ = update_and_normalize_observations(stats, obs)
        obs = OLT(observation=jnp.array(x2), legal_actions=[1], terminal=[0.0])
        stats, _ = update_and_normalize_observations(stats, obs)
        obs = OLT(observation=jnp.array(x3), legal_actions=[1], terminal=[0.0])
        stats, _ = update_and_normalize_observations(stats, obs)

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
    stats = dict(mean=jnp.array([0.2]), var=jnp.array([1.5]), count=jnp.array([15]))
    obs = normalize_observations(stats, obs)
    obs_norm = jnp.array(obs.observation)

    x_norm = normalize(stats, jnp.array(x))

    assert jnp.allclose(x_norm, obs_norm)
