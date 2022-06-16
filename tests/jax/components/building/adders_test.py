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
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from mava.components.jax.executing.base import ExecutorInit
import pytest
from acme import specs
from mava.components.jax.building.adders import (
    ParallelSequenceAdder,
    ParallelTransitionAdder,
    ParallelSequenceAdderConfig,
    ParallelTransitionAdderConfig,
    ParallelTransitionAdderSignature,
    ParallelSequenceAdderSignature,
    UniformAdderPriority,
)
from mava.adders import reverb as reverb_adders
from mava.components.jax import Component, building
from mava.components.jax.building import adders
from mava.components.jax.building.base import SystemInit
from mava.components.jax.building.environments import EnvironmentSpec
from mava.specs import DesignSpec, MAEnvironmentSpec
from mava.systems.jax.system import System
from mava.utils.wrapper_utils import parameterized_restart
from tests.jax.mocks import (
    MockExecutorEnvironmentLoop,
    MockOnPolicyDataServer,
    MockDataServer,
    MockDistributor,
    MockLogger,
)
from mava.systems.jax.builder import Builder


def make_fake_env_specs() -> MAEnvironmentSpec:
    """_summary_

    Returns:
        _description_
    """


@pytest.fixture
def test_system_parallel_sequence_adder() -> System:
    """Dummy system with zero components."""
    return TestSystemWithParallelSequenceAdder()


@pytest.fixture
def mock_builder() -> Builder:
    """Mock builder component.
    Returns:
        Builder
    """
    builder = Builder(components=[])
    # store
    adder = "paralleladder"
    store = SimpleNamespace(
        table_network_config={"table_0": "network_0"},
        unique_net_keys=["network_0"],
        data_server_client=MockDataServer,
    )
    builder.store = store
    return builder


@pytest.fixture
def parallel_sequence_adder() -> ParallelSequenceAdder:
    """Creates an MAPG loss fixture with trust region and clipping"""

    adder = ParallelSequenceAdder(config=ParallelSequenceAdderConfig())

    return adder


@pytest.fixture
def parallel_transition_adder() -> ParallelTransitionAdder:
    """Creates an MAPG loss fixture with trust region and clipping"""

    adder = ParallelTransitionAdder(config=ParallelTransitionAdderConfig())

    return adder


@pytest.fixture
def parallel_sequence_adder_signature() -> ParallelSequenceAdderSignature:
    """Creates an MAPG loss fixture with trust region and clipping"""

    signature = ParallelSequenceAdderSignature()

    return signature


@pytest.fixture
def parallel_transition_adder_signature() -> ParallelTransitionAdderSignature:
    """Creates an MAPG loss fixture with trust region and clipping"""

    signature = ParallelTransitionAdderSignature()

    return signature


@pytest.fixture
def uniform_priority() -> UniformAdderPriority:
    """Creates an MAPG loss fixture with trust region and clipping"""

    priority = UniformAdderPriority()

    return priority


def test_sequence_adders(
    mock_builder: Builder,
    parallel_sequence_adder: ParallelSequenceAdder,
) -> None:
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
    parallel_sequence_adder_signature.on_building_data_server_adder_signature(
        builder=mock_builder
    )
    assert mock_builder.store.adder_signature_fn


def test_transition_adders(
    mock_builder: Builder,
    parallel_transition_adder: ParallelTransitionAdder,
) -> None:
    parallel_transition_adder.on_building_executor_adder(builder=mock_builder)
    assert type(mock_builder.store.adder) == reverb_adders.ParallelNStepTransitionAdder


def test_transition_adders_signature(
    mock_builder: Builder,
    parallel_transition_adder_signature: ParallelTransitionAdderSignature,
) -> None:
    parallel_transition_adder_signature.on_building_data_server_adder_signature(
        builder=mock_builder
    )
    assert mock_builder.store.adder_signature_fn


def test_uniform_priority(
    mock_builder: Builder,
    uniform_priority: UniformAdderPriority,
) -> None:
    uniform_priority.on_building_executor_adder_priority(builder=mock_builder)
    assert mock_builder.store.priority_fns


# TEST SIGNATURES
# TODO (Kale-ab): test adder behaviour in more detail
