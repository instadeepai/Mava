from typing import Any, Callable, Dict, List, Sequence

import pytest

from mava.components.jax.building import DefaultNetworks
from mava.components.jax.building.networks import Networks, NetworksConfig
from mava.core_jax import SystemBuilder
from mava.specs import MAEnvironmentSpec
from mava.systems.jax import Builder, mappo
from tests.jax.mocks import make_fake_env_specs


class TestDefaultNetworks(DefaultNetworks):
    def __init__(self, test_network_factory: Callable):
        networks_config = NetworksConfig()
        networks_config.network_factory = test_network_factory
        networks_config.seed = 919

        # Normally set in the system_init
        networks_config.shared_weights = True  # type: ignore

        super().__init__(networks_config)


@pytest.fixture
def test_mappo_network_factory() -> Callable:
    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return mappo.make_default_networks(  # type: ignore
            policy_layer_sizes=(254, 254, 254),
            critic_layer_sizes=(512, 512, 256),
            *args,
            **kwargs,
        )

    return network_factory


@pytest.fixture
def test_network_factory() -> Callable:
    def make_default_networks(
        environment_spec: MAEnvironmentSpec,
        agent_net_keys: Dict[str, str],
        rng_key: List[int],
        net_spec_keys: Dict[str, str] = {},
        policy_layer_sizes: Sequence[int] = (
            256,
            256,
            256,
        ),
        critic_layer_sizes: Sequence[int] = (512, 512, 256),
    ) -> Dict[str, Any]:
        net_keys = {"1", "2", "3"}
        networks = {}

        for net_key in net_keys:
            networks[net_key] = {
                "environment_spec": environment_spec,
                "agent_net_keys": agent_net_keys,
                "rng_key": rng_key,
                "net_spec_keys": net_spec_keys,
                "policy_layer_sizes": policy_layer_sizes,
                "critic_layer_sizes": critic_layer_sizes,
                "net_key": net_key,
            }

        return networks

    def network_factory(*args: Any, **kwargs: Any) -> Any:
        return make_default_networks(  # type: ignore
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
    test_builder = Builder(components=[])
    test_builder.store.environment_spec = make_fake_env_specs()
    test_builder.store.agent_net_keys = {"key1": "value1", "key2": "value2"}
    return test_builder


@pytest.fixture
def test_default_networks(test_network_factory: Callable) -> Networks:
    return TestDefaultNetworks(test_network_factory)


def test_assert_true(
    test_default_networks: Networks, test_builder: SystemBuilder
) -> None:
    test_default_networks.on_building_init_start(test_builder)
    networks = test_builder.store.network_factory()
    print(networks)

    assert True
