# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for TransitionDataset and TrajectoryDataset classes for Jax-based Mava systems"""

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable, Dict, Optional

import pytest
import reverb
from tensorflow.python.framework import dtypes, ops

from mava import specs
from mava.adders import reverb as reverb_adders
from mava.components.jax.building.datasets import TrajectoryDataset, TransitionDataset
from mava.systems.jax.builder import Builder
from tests.jax.mocks import make_fake_env_specs

env_spec = make_fake_env_specs()
Transform = Callable[[reverb.ReplaySample], reverb.ReplaySample]


def adder_signature_fn(
    ma_environment_spec: specs.MAEnvironmentSpec,
    extras_specs: Dict[str, Any],
) -> Any:
    """signature function that helps in building a simple server"""
    return reverb_adders.ParallelNStepTransitionAdder.signature(
        ma_environment_spec=ma_environment_spec, extras_specs=extras_specs
    )


class MockBuilder(Builder):
    """Mock Builder component"""

    def __init__(self) -> None:
        self.simple_server = reverb.Server(
            [
                reverb.Table.queue(
                    name="table_0",
                    max_size=100,
                    signature=adder_signature_fn(env_spec, {}),
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
    """Create builder mock"""
    return MockBuilder()


@dataclass
class TransitionDatasetConfigTest:
    sample_batch_size: int = 512
    prefetch_size: Optional[int] = None
    num_parallel_calls: int = 24
    max_in_flight_samples_per_worker: Optional[int] = None
    postprocess: Optional[Transform] = None


@dataclass
class TrajectoryDatasetConfigTest:
    sample_batch_size: int = 512
    max_in_flight_samples_per_worker: int = 1024
    num_workers_per_iterator: int = -2
    max_samples_per_stream: int = -2
    rate_limiter_timeout_ms: int = -2
    get_signature_timeout_secs: Optional[int] = None


def test_init_transition_dataset() -> None:
    """Test initiator of TransitionDataset component"""
    transition_dataset = TransitionDataset(config=TransitionDatasetConfigTest)

    assert transition_dataset.config.sample_batch_size == 512
    assert transition_dataset.config.prefetch_size == None
    assert transition_dataset.config.num_parallel_calls == 24
    assert transition_dataset.config.max_in_flight_samples_per_worker == None
    assert transition_dataset.config.postprocess == None


def test_on_building_trainer_dataset_transition_dataset(
    mock_builder: MockBuilder,
) -> None:
    """Test on_building_trainer_dataset of TransitionDataset Component

    Args:
        mock_builder: Builder
    """
    transition_dataset = TransitionDataset()
    transition_dataset.on_building_trainer_dataset(builder=mock_builder)

    dataset = mock_builder.store.dataset._dataset._map_func._func(1)._dataset
    assert (
        dataset._input_dataset._server_address
        == mock_builder.store.data_server_client.server_address
    )
    assert dataset._input_dataset._table == mock_builder.store.trainer_id
    assert dataset._batch_size == transition_dataset.config.sample_batch_size
    assert (
        dataset._input_dataset._max_in_flight_samples_per_worker
        == 2 * transition_dataset.config.sample_batch_size
    )
    assert (
        mock_builder.store.dataset._dataset._num_parallel_calls
        == ops.convert_to_tensor(
            transition_dataset.config.num_parallel_calls,
            dtype=dtypes.int64,
            name="num_parallel_calls",
        )
    )


def test_init_trajectory_dataset() -> None:
    """Test initiator of TrajectoryDataset component"""
    trajectory_dataset = TrajectoryDataset(config=TrajectoryDatasetConfigTest)

    assert trajectory_dataset.config.sample_batch_size == 512
    assert trajectory_dataset.config.max_in_flight_samples_per_worker == 1024
    assert trajectory_dataset.config.num_workers_per_iterator == -2
    assert trajectory_dataset.config.max_samples_per_stream == -2
    assert trajectory_dataset.config.rate_limiter_timeout_ms == -2
    assert trajectory_dataset.config.get_signature_timeout_secs == None


def test_on_building_trainer_dataset_trajectory_dataset(
    mock_builder: MockBuilder,
) -> None:
    """Test on_building_trainer_dataset of TrajectoryDataset Component

    Args:
        mock_builder: Builder
    """
    trajectory_dataset = TrajectoryDataset()
    trajectory_dataset.on_building_trainer_dataset(builder=mock_builder)

    assert (
        mock_builder.store.sample_batch_size
        == trajectory_dataset.config.sample_batch_size
    )

    dataset = mock_builder.store.dataset_iterator._iterator._dataset
    assert (
        dataset._input_dataset._server_address
        == mock_builder.store.data_server_client.server_address
    )
    assert dataset._input_dataset._table == mock_builder.store.trainer_id
    assert (
        dataset._input_dataset._max_in_flight_samples_per_worker
        == 2 * trajectory_dataset.config.sample_batch_size
    )
    assert (
        dataset._input_dataset._num_workers_per_iterator
        == trajectory_dataset.config.num_workers_per_iterator
    )
    assert (
        dataset._input_dataset._max_samples_per_stream
        == trajectory_dataset.config.max_samples_per_stream
    )
    assert (
        dataset._input_dataset._rate_limiter_timeout_ms
        == trajectory_dataset.config.rate_limiter_timeout_ms
    )
