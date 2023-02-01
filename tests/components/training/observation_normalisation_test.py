from types import SimpleNamespace

import numpy as np
import pytest
import tree

from mava import constants
from mava.components.normalisation.observation_normalisation import (
    ObservationNormalisation,
    ObservationNormalisationConfig,
)
from mava.systems.builder import Builder
from mava.utils.jax_training_utils import init_norm_params
from tests.mocks import make_fake_env_specs


class MockCoreComponent(Builder):
    def __init__(self, store: SimpleNamespace) -> None:
        """Creates MockCoreComponent"""
        self.store = store


@pytest.fixture
def trainer() -> MockCoreComponent:
    """Creates a mock trainer"""
    store = SimpleNamespace(
        obs_normalisation_start=0,
        norm_params={
            constants.OBS_NORM_STATE_DICT_KEY: {"params": {"mean": np.zeros(1)}}
        },
    )
    return MockCoreComponent(store)


@pytest.fixture
def builder() -> MockCoreComponent:
    """Creates a mock builder"""
    store = SimpleNamespace(
        agents=["agent_0", "agent_1"], ma_environment_spec=make_fake_env_specs()
    )
    return MockCoreComponent(store)


@pytest.fixture
def server() -> MockCoreComponent:
    """Creates a mock server"""
    store = SimpleNamespace(parameters={})
    return MockCoreComponent(store)


@pytest.fixture
def obs_normaliser() -> ObservationNormalisation:
    """Creates mock observation normalisation component"""
    return ObservationNormalisation(
        ObservationNormalisationConfig(normalise_observations=True)
    )


def test_on_building_init(
    obs_normaliser: ObservationNormalisation, builder: Builder
) -> None:
    """Test that norm params are initialised correctly"""
    obs_normaliser.on_building_init(builder)
    assert builder.store.norm_params == {}


def test_on_building_init_end(
    obs_normaliser: ObservationNormalisation, builder: Builder
) -> None:
    """Tests that observation normalisation parameters are created correctly"""
    obs_normaliser.on_building_init(builder)
    obs_normaliser.on_building_init_end(builder)

    expected_norm_params = {
        "obs_norm_params": {
            "agent_0": init_norm_params((10,)),
            "agent_1": init_norm_params((10,)),
        }
    }

    assert tree.map_structure(
        lambda x, y: x == y, builder.store.norm_params, expected_norm_params
    )


def test_on_training_utility_fns(
    obs_normaliser: ObservationNormalisation, trainer: MockCoreComponent
) -> None:
    """Tests that trainer creates norm_obs_running_stats_fn"""
    obs_normaliser.on_training_utility_fns(trainer)  # type: ignore
    assert hasattr(trainer.store, "norm_obs_running_stats_fn")


def test_on_parameter_server_init(
    obs_normaliser: ObservationNormalisation, server: MockCoreComponent
) -> None:
    """Test that parameter get correctly added to the server"""
    norm_params = {"params": [1, 2, 3, 4]}
    server.store.norm_params = norm_params
    server.store.parameters = {"test": 1}

    obs_normaliser.on_parameter_server_init(server)  # type: ignore

    assert server.store.parameters["norm_params"] == norm_params
    # make sure we don't overwrite parameters
    assert server.store.parameters["test"] == 1
