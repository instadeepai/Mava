import jax.numpy as jnp
import numpy as np

from mava.utils.jax_training_utils import (
    compute_running_mean_var_count,
    denormalize,
    normalize,
)


def test_compute_running_mean_var_count() -> None:
    """Test if the running mean, variance and data counts are computed correctly"""

    for (x1, x2, x3) in [
        (np.random.randn(6), np.random.randn(8), np.random.randn(10)),
        (np.random.randn(10, 1), np.random.randn(15, 1), np.random.randn(20, 1)),
    ]:

        stats = jnp.array([0, 0, 1e-4])

        stats = compute_running_mean_var_count(stats, jnp.array(x1))
        stats = compute_running_mean_var_count(stats, jnp.array(x2))
        stats = compute_running_mean_var_count(stats, jnp.array(x3))

        x = jnp.array(np.concatenate([x1, x2, x3], axis=0))
        stats2 = jnp.array([jnp.mean(x), jnp.var(x), x.size])

        assert jnp.allclose(stats, stats2)


def test_normalization() -> None:
    """Test if the normalization does the right thing"""

    x = jnp.array(np.random.randn(15))
    stats = jnp.array([0.2, 1.5, 15])
    x_norm = normalize(stats, x)
    x_denorm = denormalize(stats, x_norm)

    assert jnp.allclose(x, x_denorm)
