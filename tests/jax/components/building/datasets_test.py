from typing import Dict, Any
from types import SimpleNamespace

import pytest
import reverb
from reverb import item_selectors, rate_limiters

from mava.components.jax.building.datasets import TransitionDataset
from mava.systems.jax.builder import Builder
from mava import specs
from mava.adders import reverb as reverb_adders

from tests.jax.mocks import make_fake_env_specs
env_spec = make_fake_env_specs()

def adder_signature_fn(
            ma_environment_spec: specs.MAEnvironmentSpec,
            extras_specs: Dict[str, Any],
        ) -> Any:
            return reverb_adders.ParallelNStepTransitionAdder.signature(
                ma_environment_spec=ma_environment_spec, extras_specs=extras_specs
            )
class MockBuilder(Builder):
    def __init__(self) -> None:
        self.simple_server = reverb.Server(
            [
                reverb.Table.queue(
                    name="table_0",
                    max_size=100,
                    signature=adder_signature_fn(
                        env_spec, {}
                    ),
                )
            ]
        )
        data_server_client = SimpleNamespace(
            server_address=f"localhost:{self.simple_server.port}"
        )
        trainer_id = "table_0"
        self.store = SimpleNamespace(
            data_server_client=data_server_client, trainer_id=trainer_id
        )


@pytest.fixture
def mock_builder() -> MockBuilder:
    return MockBuilder()


def test_init_transition_dataset() -> None:
    transition_dataset = TransitionDataset()

    assert transition_dataset.config.sample_batch_size == 256
    assert transition_dataset.config.prefetch_size == None
    assert transition_dataset.config.num_parallel_calls == 12
    assert transition_dataset.config.max_in_flight_samples_per_worker == None
    assert transition_dataset.config.postprocess == None


def test_on_building_trainer_dataset(mock_builder: MockBuilder):
    transition_dataset = TransitionDataset()
    transition_dataset.on_building_trainer_dataset(builder=mock_builder)
