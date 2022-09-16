import pytest
import reverb

from mava.components.building.reverb_components import (
    MinSizeRateLimiter,
    RateLimiterConfig,
    SampleToInsertRateLimiter,
)
from mava.core_jax import SystemBuilder
from mava.systems import Builder


@pytest.fixture
def rate_limiter_config() -> RateLimiterConfig:
    """Fixture for manually created RateLimiterConfig."""
    return RateLimiterConfig(
        min_data_server_size=100, samples_per_insert=16.0, error_buffer=500.0
    )


@pytest.fixture
def min_size_rate_limiter(rate_limiter_config: RateLimiterConfig) -> MinSizeRateLimiter:
    """Fixture for basic MinSizeRateLimiter."""
    return MinSizeRateLimiter(config=rate_limiter_config)


@pytest.fixture
def sample_to_insert_rate_limiter(
    rate_limiter_config: RateLimiterConfig,
) -> SampleToInsertRateLimiter:
    """Fixture for basic SampleToInsertRateLimiter."""
    return SampleToInsertRateLimiter(config=rate_limiter_config)


@pytest.fixture
def builder() -> SystemBuilder:
    """Pytest fixture for system builder."""
    system_builder = Builder(components=[])
    return system_builder


def test_min_size_rate_limiter(
    builder: Builder, min_size_rate_limiter: MinSizeRateLimiter
) -> None:
    """Test MinSizeRateLimiter by manually calling the hook."""
    # Config loaded
    assert min_size_rate_limiter.config.min_data_server_size == 100
    assert min_size_rate_limiter.config.samples_per_insert == 16.0
    assert min_size_rate_limiter.config.error_buffer == 500.0

    # Rate limiter function created by hook
    assert not hasattr(builder.store, "rate_limiter_fn")
    min_size_rate_limiter.on_building_data_server_rate_limiter(builder)

    reverb_rate_limiter = builder.store.rate_limiter_fn()
    assert isinstance(reverb_rate_limiter, reverb.rate_limiters.MinSize)

    # TODO: remove below when following test is no longer skipped
    assert repr(reverb_rate_limiter).split("min_size_to_sample=")[1][:3] == "100"


# TODO: do not skip this test when reverb is upgraded to 0.8.0
@pytest.mark.skip
def test_min_size_rate_limiter_attributes(
    builder: Builder, min_size_rate_limiter: MinSizeRateLimiter
) -> None:
    """Test MinSizeRateLimiter attributes loaded correctly."""
    min_size_rate_limiter.on_building_data_server_rate_limiter(builder)
    reverb_rate_limiter = builder.store.rate_limiter_fn()
    assert reverb_rate_limiter._min_size_to_sample == 100


def test_sample_to_insert_rate_limiter_with_error_buffer(
    builder: SystemBuilder, sample_to_insert_rate_limiter: SampleToInsertRateLimiter
) -> None:
    """Test SampleToInsertRateLimiter when an error buffer is provided."""
    # Config loaded
    assert sample_to_insert_rate_limiter.config.min_data_server_size == 100
    assert sample_to_insert_rate_limiter.config.samples_per_insert == 16.0
    assert sample_to_insert_rate_limiter.config.error_buffer == 500.0

    # Rate limiter function created by hook
    assert not hasattr(builder.store, "rate_limiter_fn")
    sample_to_insert_rate_limiter.on_building_data_server_rate_limiter(builder)

    reverb_rate_limiter = builder.store.rate_limiter_fn()
    assert isinstance(reverb_rate_limiter, reverb.rate_limiters.SampleToInsertRatio)

    # TODO: remove below when following test is no longer skipped

    assert repr(reverb_rate_limiter).split("min_size_to_sample=")[1][:3] == "100"
    assert repr(reverb_rate_limiter).split("samples_per_insert=")[1][:2] == "16"

    # Ensure offset created correctly
    offset = (
        16  # reverb_rate_limiter._samples_per_insert
        * 100  # reverb_rate_limiter._min_size_to_sample
    )
    min_diff = offset - sample_to_insert_rate_limiter.config.error_buffer
    max_diff = offset + sample_to_insert_rate_limiter.config.error_buffer

    assert int(repr(reverb_rate_limiter).split("min_diff_=")[1][:4]) == int(min_diff)
    assert int(repr(reverb_rate_limiter).split("max_diff=")[1][:4]) == int(max_diff)


# TODO: do not skip this test when reverb is upgraded to 0.8.0
@pytest.mark.skip
def test_sample_to_insert_rate_limiter_with_error_buffer_attributes(
    builder: SystemBuilder, sample_to_insert_rate_limiter: SampleToInsertRateLimiter
) -> None:
    """Test SampleToInsertRateLimiter attributes loaded correctly with error buffer."""
    sample_to_insert_rate_limiter.on_building_data_server_rate_limiter(builder)
    reverb_rate_limiter = builder.store.rate_limiter_fn()

    assert reverb_rate_limiter._min_size_to_sample == 100
    assert reverb_rate_limiter._samples_per_insert == 16.0

    # Ensure offset created correctly
    offset = (
        reverb_rate_limiter._samples_per_insert
        * reverb_rate_limiter._min_size_to_sample
    )
    min_diff = offset - sample_to_insert_rate_limiter.config.error_buffer
    max_diff = offset + sample_to_insert_rate_limiter.config.error_buffer
    assert reverb_rate_limiter._min_diff == min_diff
    assert reverb_rate_limiter._max_diff == max_diff


def test_sample_to_insert_rate_limiter_no_error_buffer(
    builder: SystemBuilder, sample_to_insert_rate_limiter: SampleToInsertRateLimiter
) -> None:
    """Test SampleToInsertRateLimiter when no error buffer is provided."""
    # Manually set config
    sample_to_insert_rate_limiter.config.error_buffer = None

    # Rate limiter function created by hook
    assert not hasattr(builder.store, "rate_limiter_fn")
    sample_to_insert_rate_limiter.on_building_data_server_rate_limiter(builder)

    reverb_rate_limiter = builder.store.rate_limiter_fn()
    assert isinstance(reverb_rate_limiter, reverb.rate_limiters.SampleToInsertRatio)

    # TODO: remove below when following test is no longer skipped

    assert repr(reverb_rate_limiter).split("min_size_to_sample=")[1][:3] == "100"
    assert repr(reverb_rate_limiter).split("samples_per_insert=")[1][:2] == "16"

    # Ensure offset created correctly
    samples_per_insert_tolerance = (
        0.1 * sample_to_insert_rate_limiter.config.samples_per_insert
    )
    error_buffer = (
        sample_to_insert_rate_limiter.config.min_data_server_size
        * samples_per_insert_tolerance
    )

    offset = (
        16  # reverb_rate_limiter._samples_per_insert
        * 100  # reverb_rate_limiter._min_size_to_sample
    )
    min_diff = offset - error_buffer
    max_diff = offset + error_buffer

    assert int(repr(reverb_rate_limiter).split("min_diff_=")[1][:4]) == int(min_diff)
    assert int(repr(reverb_rate_limiter).split("max_diff=")[1][:4]) == int(max_diff)


# TODO: do not skip this test when reverb is upgraded to 0.8.0
@pytest.mark.skip
def test_sample_to_insert_rate_limiter_no_error_buffer_attributes(
    builder: SystemBuilder, sample_to_insert_rate_limiter: SampleToInsertRateLimiter
) -> None:
    """Test SampleToInsertRateLimiter attributes without error buffer."""
    # Manually set config
    sample_to_insert_rate_limiter.config.error_buffer = None
    sample_to_insert_rate_limiter.on_building_data_server_rate_limiter(builder)

    reverb_rate_limiter = builder.store.rate_limiter_fn()
    assert reverb_rate_limiter._min_size_to_sample == 100
    assert reverb_rate_limiter._samples_per_insert == 16.0

    # Ensure offset created correctly
    samples_per_insert_tolerance = (
        0.1 * sample_to_insert_rate_limiter.config.samples_per_insert
    )
    error_buffer = (
        sample_to_insert_rate_limiter.config.min_data_server_size
        * samples_per_insert_tolerance
    )

    offset = (
        reverb_rate_limiter._samples_per_insert
        * reverb_rate_limiter._min_size_to_sample
    )
    min_diff = offset - error_buffer
    max_diff = offset + error_buffer
    assert reverb_rate_limiter._min_diff == min_diff
    assert reverb_rate_limiter._max_diff == max_diff
