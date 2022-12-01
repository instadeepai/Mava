from types import SimpleNamespace

import pytest
from acme.specs import DiscreteArray

from mava.core_jax import SystemBuilder
from mava.systems.builder import Builder
from mava.systems.idqn.components.building.extras_spec import DQNExtrasSpec
from tests.mocks import make_fake_env_specs


class MockBuilder(Builder):
    def __init__(self, store: SimpleNamespace) -> None:
        """Initialises a mock builder"""
        self.store = store


@pytest.fixture
def builder() -> Builder:
    """Creates a mock builder"""
    return MockBuilder(
        SimpleNamespace(
            unique_net_keys=[1, 2],
            ma_environment_spec=make_fake_env_specs(),
        )
    )


def test_on_building_init_end(builder: SystemBuilder) -> None:
    """Tests that extras spec is created in the store and only network keys are added"""
    extras_spec_component = DQNExtrasSpec()
    extras_spec_component.on_building_init_end(builder)
    assert builder.store.extras_spec == {
        "network_keys": {"agent_0": DiscreteArray(2), "agent_1": DiscreteArray(2)}
    }
