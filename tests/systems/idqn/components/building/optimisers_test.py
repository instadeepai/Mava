import optax
import pytest

from mava.core_jax import SystemBuilder
from mava.systems.builder import Builder
from mava.systems.idqn.components.building.optimisers import Optimiser, OptimisersConfig


@pytest.fixture
def builder() -> Builder:
    """Creates a mock builder"""
    return Builder([])


def test_on_building_init_start_with_optimiser(builder: SystemBuilder) -> None:
    """Tests that if an optimiser is passed that it will be used"""
    optimiser = optax.sgd(1)

    optimiser_componet = Optimiser(OptimisersConfig(policy_optimiser=optimiser))
    optimiser_componet.on_building_init_start(builder)

    assert builder.store.policy_optimiser == optimiser


def test_on_building_init_start_no_config(builder: SystemBuilder) -> None:
    """Tests that by default optimisers create a GradientTransformation in the store"""
    optimiser_componet = Optimiser()
    optimiser_componet.on_building_init_start(builder)

    # cannot test exact values, because builder.store.policy_optimiser is a function
    assert isinstance(builder.store.policy_optimiser, optax.GradientTransformation)
