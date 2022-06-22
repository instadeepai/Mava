import pytest

from mava.components.jax.building.rate_limiters import MinSizeRateLimiter, RateLimiterConfig


@pytest.fixture
def rate_limiter_config() -> RateLimiterConfig:
    return RateLimiterConfig(min_data_server_size=100, samples_per_insert=16.0, error_buffer=5.0)


@pytest.fixture
def min_size_rate_limiter(rate_limiter_config) -> MinSizeRateLimiter:
    return MinSizeRateLimiter(config=rate_limiter_config)


@pytest.fixture
def min_size_rate_limiter(rate_limiter_config) -> MinSizeRateLimiter:
    return MinSizeRateLimiter(config=rate_limiter_config)
