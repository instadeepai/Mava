import functools

import pytest

from mava.components.jax.building.environments import (
    EnvironmentSpec,
    EnvironmentSpecConfig,
)
from mava.core_jax import SystemBuilder
from mava.systems.jax import Builder
from mava.utils.environments import debugging_utils


@pytest.fixture
def test_environment_spec() -> EnvironmentSpec:
    """Pytest fixture for environment spec"""
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
    )

    config = EnvironmentSpecConfig(environment_factory=environment_factory)
    test_environment_spec = EnvironmentSpec(config=config)
    return test_environment_spec


@pytest.fixture
def test_builder() -> SystemBuilder:
    """Pytest fixture for system builder."""
    system_builder = Builder(components=[])
    return system_builder
