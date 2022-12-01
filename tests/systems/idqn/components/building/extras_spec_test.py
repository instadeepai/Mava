from types import SimpleNamespace

import numpy as np
import pytest
from acme.specs import DiscreteArray

from mava.core_jax import SystemBuilder
from mava.systems.builder import Builder
from mava.systems.idqn.components.building.extras_spec import DQNExtrasSpec
from tests.mocks import make_fake_env_specs


class MockBuilder(Builder):
    def __init__(self, store) -> None:
        self.store = store


@pytest.fixture
def builder():
    return MockBuilder(
        SimpleNamespace(
            unique_net_keys=["network_0", "network_1"],
            ma_environment_spec=make_fake_env_specs(),
        )
    )


def test_on_building_init_end(builder: SystemBuilder):
    """Tests that extras spec is created"""
    extras_spec_component = DQNExtrasSpec()
    extras_spec_component.on_building_init_end(builder)
    assert builder.store.extras_spec == {
        "network_keys": {"agent_0": DiscreteArray(2), "agent_1": DiscreteArray(2)}
    }
