import pytest
import reverb

from mava.components.jax.building.rate_limiters import (
    MinSizeRateLimiter,
    RateLimiterConfig,
    SampleToInsertRateLimiter,
)
from mava.core_jax import SystemBuilder
from mava.systems.jax import Builder


@pytest.fixture
def rate_limiter_config() -> RateLimiterConfig:
    return RateLimiterConfig(
        min_data_server_size=100, samples_per_insert=16.0, error_buffer=5.0
    )


@pytest.fixture
def min_size_rate_limiter(rate_limiter_config) -> MinSizeRateLimiter:
    return MinSizeRateLimiter(config=rate_limiter_config)


@pytest.fixture
def sample_to_insert_rate_limiter(rate_limiter_config) -> SampleToInsertRateLimiter:
    return SampleToInsertRateLimiter(config=rate_limiter_config)


@pytest.fixture
def builder() -> SystemBuilder:
    """Pytest fixture for system builder.

    Returns:
        System builder with no components.
    """
    system_builder = Builder(components=[])
    return system_builder


def test_min_size_rate_limiter(builder, min_size_rate_limiter) -> None:
    # Config loaded
    assert min_size_rate_limiter.config.min_data_server_size == 100
    assert min_size_rate_limiter.config.samples_per_insert == 16.0
    assert min_size_rate_limiter.config.error_buffer == 5.0

    # Rate limiter function created by hook
    assert not hasattr(builder.store, "rate_limiter_fn")
    min_size_rate_limiter.on_building_data_server_rate_limiter(builder)

    rate_limiter = builder.store.rate_limiter_fn()
    assert isinstance(rate_limiter, reverb.rate_limiters.MinSize)
    # TODO: the below should pass, but it does not
    # See https://github.com/deepmind/reverb/blob/7c33ea44589deb5c5ac440cb1b3a89319241b56e/reverb/rate_limiters.py#L31
    # assert rate_limiter._min_size_to_sample == 100
