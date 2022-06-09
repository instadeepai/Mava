from typing import Any, Callable

import pytest

from mava.components.jax.building import DefaultNetworks
from mava.components.jax.building.networks import Networks, NetworksConfig
from mava.core_jax import SystemBuilder
from mava.systems.jax import Builder, mappo


class TestDefaultNetworks(DefaultNetworks):
    def __init__(self, test_network_factory: Callable):
        networks_config = NetworksConfig()
        networks_config.network_factory = test_network_factory
        networks_config.seed = 919

        super().__init__(networks_config)


@pytest.fixture
def test_network_factory() -> Callable:
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return mappo.make_default_networks(  # type: ignore
            policy_layer_sizes=(254, 254, 254),
            critic_layer_sizes=(512, 512, 256),
            *args,
            **kwargs,
        )

    return network_factory


@pytest.fixture
def test_builder() -> SystemBuilder:
    """Pytest fixture for system builder. Adds executor and trainer IDs to the store.
    Returns:
        System builder with no components.
    """
    return Builder(components=[])


@pytest.fixture
def test_default_networks(test_network_factory: Callable) -> Networks:
    return TestDefaultNetworks(test_network_factory)


def test_assert_true(
    test_default_networks: Networks, test_builder: SystemBuilder
) -> None:
    assert True
