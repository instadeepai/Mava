from types import SimpleNamespace
import pytest
from mava.components.jax.building.datasets import TransitionDataset
from mava.systems.jax.builder import Builder
from reverb import item_selectors, rate_limiters
import reverb

class MockBuilder(Builder):
    def __init__(self) -> None:
        simple_server=reverb.Server(tables=[reverb.Table(
                name="table_0",
                sampler=item_selectors.Prioritized(priority_exponent=1),
                remover=item_selectors.Fifo(),
                max_size=1000,
                rate_limiter=rate_limiters.MinSize(1),
            )
        ],port=2000)
        """ https://colab.research.google.com/github/deepmind/reverb/blob/master/examples/demo.ipynb#scrollTo=SQFSZJkyroFX """
        data_server_client=SimpleNamespace(server_address=f'localhost:{simple_server.port}')
        trainer_id= "table_0"
        self.store = SimpleNamespace(
            data_server_client=data_server_client,
            trainer_id=trainer_id)
        

@pytest.fixture
def mock_builder()->MockBuilder:
    return MockBuilder()

def test_init_transition_dataset()->None:
    transition_dataset=TransitionDataset()

    assert transition_dataset.config.sample_batch_size== 256
    assert transition_dataset.config.prefetch_size==None
    assert transition_dataset.config.num_parallel_calls==12
    assert transition_dataset.config.max_in_flight_samples_per_worker==None
    assert transition_dataset.config.postprocess==None

def test_on_building_trainer_dataset(mock_builder:MockBuilder):
    transition_dataset=TransitionDataset()
    transition_dataset.on_building_trainer_dataset(builder=mock_builder)


