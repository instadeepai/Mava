from types import SimpleNamespace

import pytest
import reverb
from reverb import item_selectors, rate_limiters

from mava.components.jax.building.adders import ParallelTransitionAdderSignature
from mava.components.jax.building.datasets import TransitionDataset
from mava.systems.jax.builder import Builder
from tests.jax.mocks import MockDataServer, MockOnPolicyDataServer, make_fake_env_specs


class MockBuilder(Builder):
    def __init__(self) -> None:
        simple_server = reverb.Server(
            tables=[
                reverb.Table(
                    name="table_0",
                    sampler=item_selectors.Prioritized(priority_exponent=1),
                    remover=item_selectors.Fifo(),
                    max_size=1000,
                    rate_limiter=rate_limiters.MinSize(1),
                )
            ],
            port=None,
        )
        agent_net_keys = {
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
        }
        data_server_client = MockOnPolicyDataServer
        data_server_client.server_address = f"localhost:{simple_server.port}"
        trainer_id = "table_0"
        ma_environment_spec = make_fake_env_specs()
        self.store = SimpleNamespace(
            data_server_client=data_server_client,
            trainer_id=trainer_id,
            ma_environment_spec=ma_environment_spec,
            agent_net_keys=agent_net_keys,
        )


@pytest.fixture
def mock_builder() -> MockBuilder:
    return MockBuilder()


@pytest.fixture
def parallel_transition_adder_signature() -> ParallelTransitionAdderSignature:
    return ParallelTransitionAdderSignature()


@pytest.fixture
def mock_data_server() -> MockDataServer:
    return MockOnPolicyDataServer()


def test_init_transition_dataset() -> None:
    transition_dataset = TransitionDataset()

    assert transition_dataset.config.sample_batch_size == 256
    assert transition_dataset.config.prefetch_size is None
    assert transition_dataset.config.num_parallel_calls == 12
    assert transition_dataset.config.max_in_flight_samples_per_worker is None
    assert transition_dataset.config.postprocess is None


# TODO test for both transition and trajectory adders
def test_on_building_trainer_dataset(
    mock_builder: MockBuilder,
    parallel_transition_adder_signature: ParallelTransitionAdderSignature,
    mock_data_server: MockDataServer,
) -> None:
    transition_dataset = TransitionDataset()
    parallel_transition_adder_signature.on_building_data_server_adder_signature(
        mock_builder
    )
    mock_data_server.on_building_data_server(mock_builder)
    transition_dataset.on_building_trainer_dataset(builder=mock_builder)
