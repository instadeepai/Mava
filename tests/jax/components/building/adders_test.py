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

"""Tests for config class for Jax-based Mava systems"""

from types import SimpleNamespace

import pytest

from mava.adders import reverb as reverb_adders
from mava.components.jax.building.adders import (
    ParallelSequenceAdder,
    ParallelSequenceAdderConfig,
    ParallelSequenceAdderSignature,
    ParallelTransitionAdder,
    ParallelTransitionAdderConfig,
    ParallelTransitionAdderSignature,
    UniformAdderPriority,
)
from mava.systems.jax.builder import Builder
from tests.jax.mocks import MockDataServer


@pytest.fixture
def mock_builder() -> Builder:
    """Mock builder component.

    Returns:
        System builder with no components.
    """
    builder = Builder(components=[])
    # store
    store = SimpleNamespace(
        table_network_config={"table_0": "network_0"},
        unique_net_keys=["network_0"],
        data_server_client=MockDataServer,
    )
    builder.store = store
    return builder


@pytest.fixture
def parallel_sequence_adder() -> ParallelSequenceAdder:
    """Creates a parallel sequence adder fixture with config

    Returns:
        ParallelSequenceAdder with ParallelSequenceAdderConfig.
    """

    adder = ParallelSequenceAdder(config=ParallelSequenceAdderConfig())
    return adder


@pytest.fixture
def parallel_transition_adder() -> ParallelTransitionAdder:
    """Creates a parallel transition adder fixture with config

    Returns:
        ParallelTransitionAdder with ParallelTransitionAdderConfig.
    """

    adder = ParallelTransitionAdder(config=ParallelTransitionAdderConfig())
    return adder


@pytest.fixture
def parallel_sequence_adder_signature() -> ParallelSequenceAdderSignature:
    """Creates a paralell sequence signature fixture

    Returns:
        ParallelSequenceAdderSignature.
    """

    signature = ParallelSequenceAdderSignature()
    return signature


@pytest.fixture
def parallel_transition_adder_signature() -> ParallelTransitionAdderSignature:
    """Creates a paralell transition adder signature fixture

    Returns:
        ParallelTransitionAdderSignature.
    """

    signature = ParallelTransitionAdderSignature()
    return signature


@pytest.fixture
def uniform_priority() -> UniformAdderPriority:
    """Creates a uniform adder priority fixture

    Returns:
        UniformAdderPriority.
    """

    priority = UniformAdderPriority()

    return priority


def test_sequence_adders(
    mock_builder: Builder,
    parallel_sequence_adder: ParallelSequenceAdder,
) -> None:
    """Test sequence adder callbacks.

    Args:
        mock_builder: Fixture SystemBuilder.
        parallel_sequence_adder: Fixture ParallelSequenceAdder.

    Returns:
        None
    """

    parallel_sequence_adder.on_building_init_start(builder=mock_builder)
    assert (
        mock_builder.store.sequence_length
        == parallel_sequence_adder.config.sequence_length
    )
    parallel_sequence_adder.on_building_executor_adder(builder=mock_builder)
    assert type(mock_builder.store.adder) == reverb_adders.ParallelSequenceAdder


def test_sequence_adders_signature(
    mock_builder: Builder,
    parallel_sequence_adder_signature: ParallelSequenceAdderSignature,
) -> None:
    """Test sequence adder signature callback.

    Args:
        mock_builder: Fixture SystemBuilder.
        parallel_sequence_adder_signature: Fixture ParallelSequenceAdderSignature.

    Returns:
        None
    """
    parallel_sequence_adder_signature.on_building_data_server_adder_signature(
        builder=mock_builder
    )
    assert mock_builder.store.adder_signature_fn


def test_transition_adders(
    mock_builder: Builder,
    parallel_transition_adder: ParallelTransitionAdder,
) -> None:
    """Test transition adder callback

    Args:
        mock_builder: Fixture SystemBuilder.
        parallel_transition_adder: Fixture ParallelTransitionAdder.

    Returns:
        None
    """
    parallel_transition_adder.on_building_executor_adder(builder=mock_builder)
    assert type(mock_builder.store.adder) == reverb_adders.ParallelNStepTransitionAdder


def test_transition_adders_signature(
    mock_builder: Builder,
    parallel_transition_adder_signature: ParallelTransitionAdderSignature,
) -> None:
    """Test transition adder signature callback

    Args:
        mock_builder: Fixture SystemBuilder.
        parallel_transition_adder_signature: Fixture ParallelTransitionAdderSignature.

    Returns:
        None
    """
    parallel_transition_adder_signature.on_building_data_server_adder_signature(
        builder=mock_builder
    )
    assert mock_builder.store.adder_signature_fn


def test_uniform_priority(
    mock_builder: Builder,
    uniform_priority: UniformAdderPriority,
) -> None:
    """Test uniform priority callback

    Args:
        mock_builder: Fixture SystemBuilder.
        uniform_priority: Fixture UniformAdderPriority.

    Returns:
        None
    """
    uniform_priority.on_building_executor_adder_priority(builder=mock_builder)
    assert mock_builder.store.priority_fns
