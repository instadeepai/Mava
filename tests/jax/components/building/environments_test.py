import functools

import numpy as np
import pytest

from mava import MAEnvironmentSpec, specs
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
        num_agents=10,
    )

    config = EnvironmentSpecConfig(environment_factory=environment_factory)
    test_environment_spec = EnvironmentSpec(config=config)
    return test_environment_spec


@pytest.fixture
def test_builder() -> SystemBuilder:
    """Pytest fixture for system builder."""
    system_builder = Builder(components=[])
    return system_builder


class TestEnvironmentSpec:
    """Tests for EnvironmentSpec"""

    def test_init(self, test_environment_spec: EnvironmentSpec) -> None:
        """Test that class loads config properly"""

        environment = test_environment_spec.config.environment_factory()
        assert environment.environment.num_agents == 10

    def test_on_building_init_start(
        self, test_environment_spec: EnvironmentSpec, test_builder: SystemBuilder
    ) -> None:
        """Test by manually calling the hook and checking the store."""
        test_environment_spec.on_building_init_start(test_builder)

        # Assert for type and extra spec
        environment_spec = test_builder.store.environment_spec
        assert isinstance(environment_spec, specs.MAEnvironmentSpec)
        environment = test_environment_spec.config.environment_factory()

        # Assert correct spec created
        expected_spec = MAEnvironmentSpec(environment)
        assert environment_spec.extra_specs == expected_spec.extra_specs
        assert environment_spec._keys == expected_spec._keys
        for key in environment_spec._keys:
            assert (
                environment_spec._specs[key].observations.observation
                == expected_spec._specs[key].observations.observation
            )
            assert np.array_equal(
                environment_spec._specs[key].observations.legal_actions,
                expected_spec._specs[key].observations.legal_actions,
            )
            assert (
                environment_spec._specs[key].observations.terminal
                == expected_spec._specs[key].observations.terminal
            )
            assert (
                environment_spec._specs[key].actions
                == expected_spec._specs[key].actions
            )
            assert (
                environment_spec._specs[key].rewards
                == expected_spec._specs[key].rewards
            )
            assert (
                environment_spec._specs[key].discounts
                == expected_spec._specs[key].discounts
            )
