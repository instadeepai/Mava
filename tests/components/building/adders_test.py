# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

"""Adder unit test"""

from types import SimpleNamespace

import pytest

from mava import types
from mava.adders import reverb as reverb_adders
from mava.adders.reverb import base as reverb_base
from mava.components.building.adders import (
    ParallelSequenceAdder,
    ParallelSequenceAdderConfig,
    ParallelSequenceAdderSignature,
    ParallelTransitionAdder,
    ParallelTransitionAdderConfig,
    ParallelTransitionAdderSignature,
    UniformAdderPriority,
)
from mava.specs import MAEnvironmentSpec
from mava.systems.builder import Builder
from tests.mocks import MockDataServer, make_fake_env_specs


@pytest.fixture
def mock_builder() -> Builder:
    """Mock builder component.

    Returns:
        System builder with no components.
    """
    builder = Builder(components=[])
    store = SimpleNamespace(
        priority_fns={"table_0": 1},
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

    adder = ParallelSequenceAdder(
        config=ParallelSequenceAdderConfig(
            sequence_length=1, period=1, use_next_extras=True
        )
    )
    return adder


@pytest.fixture
def parallel_transition_adder() -> ParallelTransitionAdder:
    """Creates a parallel transition adder fixture with config

    Returns:
        ParallelTransitionAdder with ParallelTransitionAdderConfig.
    """

    adder = ParallelTransitionAdder(
        config=ParallelTransitionAdderConfig(n_step=1, discount=1)
    )
    return adder


@pytest.fixture
def parallel_sequence_adder_signature() -> ParallelSequenceAdderSignature:
    """Creates a parallel sequence signature fixture

    Returns:
        ParallelSequenceAdderSignature.
    """

    signature = ParallelSequenceAdderSignature()
    return signature


@pytest.fixture
def parallel_transition_adder_signature() -> ParallelTransitionAdderSignature:
    """Creates a parallel transition adder signature fixture

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


@pytest.fixture
def mock_env_specs() -> MAEnvironmentSpec:
    """Creates a mock environment spec

    Returns:
        MAEnvironmentSpec.
    """

    return make_fake_env_specs()


def test_parallel_sequence_adder(
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
    # ParallelSequenceAdderConfig parameters set correctly
    assert parallel_sequence_adder.config.sequence_length == 1
    assert parallel_sequence_adder.config.period == 1
    assert parallel_sequence_adder.config.use_next_extras is True

    parallel_sequence_adder.on_building_init_start(builder=mock_builder)
    parallel_sequence_adder.on_building_executor_adder(builder=mock_builder)
    assert type(mock_builder.store.adder) == reverb_adders.ParallelSequenceAdder

    # Reverb ParallelSequenceAdder args have been set correctly
    assert (
        mock_builder.store.adder._sequence_length
        == parallel_sequence_adder.config.sequence_length
    )
    assert mock_builder.store.adder._period == parallel_sequence_adder.config.period
    assert (
        mock_builder.store.adder._use_next_extras
        == parallel_sequence_adder.config.use_next_extras
    )
    assert mock_builder.store.adder._client == mock_builder.store.data_server_client
    assert mock_builder.store.adder._priority_fns == mock_builder.store.priority_fns
    assert (
        mock_builder.store.adder._net_ids_to_keys == mock_builder.store.unique_net_keys
    )
    assert (
        mock_builder.store.adder._table_network_config
        == mock_builder.store.table_network_config
    )
    assert parallel_sequence_adder.name() == "executor_adder"


def test_parallel_sequence_adder_signature(
    mock_builder: Builder,
    parallel_sequence_adder: ParallelSequenceAdder,
    parallel_sequence_adder_signature: ParallelSequenceAdderSignature,
    mock_env_specs: MAEnvironmentSpec,
) -> None:
    """Test sequence adder signature callback.

    Args:
        mock_builder: Fixture SystemBuilder.
        parallel_sequence_adder_signature: Fixture ParallelSequenceAdderSignature.
        mock_env_specs: Fixture MAEnvironmentSpec

    Returns:
        None
    """
    parallel_sequence_adder_signature.on_building_data_server_adder_signature(
        builder=mock_builder
    )

    assert hasattr(mock_builder.store, "adder_signature_fn")

    signature = mock_builder.store.adder_signature_fn(
        ma_environment_spec=mock_env_specs,
        sequence_length=parallel_sequence_adder.config.sequence_length,
        extras_specs=mock_env_specs.get_extras_specs(),
    )
    assert type(signature) == reverb_base.Step

    # Dimensions preserved after spec is generated, ignoring the time dim
    assert signature.observations["agent_0"].shape.as_list()[1:] == list(
        mock_env_specs.get_agent_environment_specs()["agent_0"].observations.shape
    )
    assert signature.observations["agent_1"].shape.as_list()[1:] == list(
        mock_env_specs.get_agent_environment_specs()["agent_1"].observations.shape
    )
    assert signature.actions["agent_1"].shape.as_list()[1:] == list(
        mock_env_specs.get_agent_environment_specs()["agent_1"].actions.shape
    )

    assert parallel_sequence_adder_signature.name() == "data_server_adder_signature"


def test_parallel_transition_adder(
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

    # ParallelTransitionAdderConfig parameters set correctly
    assert parallel_transition_adder.config.n_step == 1
    assert parallel_transition_adder.config.discount == 1
    parallel_transition_adder.on_building_executor_adder(builder=mock_builder)
    assert type(mock_builder.store.adder) == reverb_adders.ParallelNStepTransitionAdder

    # Reverb ParallelSequenceAdder args have been set correctly
    assert mock_builder.store.adder.n_step == parallel_transition_adder.config.n_step
    assert (
        mock_builder.store.adder._discount == parallel_transition_adder.config.discount
    )
    assert mock_builder.store.adder._client == mock_builder.store.data_server_client
    assert mock_builder.store.adder._priority_fns == mock_builder.store.priority_fns
    assert (
        mock_builder.store.adder._net_ids_to_keys == mock_builder.store.unique_net_keys
    )
    assert (
        mock_builder.store.adder._table_network_config
        == mock_builder.store.table_network_config
    )
    assert parallel_transition_adder.name() == "executor_adder"


def test_parallel_transition_adder_signature(
    mock_builder: Builder,
    parallel_transition_adder_signature: ParallelTransitionAdderSignature,
    mock_env_specs: MAEnvironmentSpec,
) -> None:
    """Test transition adder signature callback

    Args:
        mock_builder: Fixture SystemBuilder.
        parallel_transition_adder_signature: Fixture ParallelTransitionAdderSignature.
        mock_env_specs: Fixture MAEnvironmentSpec

    Returns:
        None
    """
    parallel_transition_adder_signature.on_building_data_server_adder_signature(
        builder=mock_builder
    )
    assert hasattr(mock_builder.store, "adder_signature_fn")

    signature = mock_builder.store.adder_signature_fn(
        ma_environment_spec=mock_env_specs,
        extras_specs=mock_env_specs.get_extras_specs(),
    )
    assert type(signature) == types.Transition

    # Dimensions preserved after spec is generated
    assert signature.observations["agent_0"].shape.as_list() == list(
        mock_env_specs.get_agent_environment_specs()["agent_0"].observations.shape
    )
    assert signature.observations["agent_1"].shape.as_list() == list(
        mock_env_specs.get_agent_environment_specs()["agent_1"].observations.shape
    )
    assert signature.actions["agent_1"].shape.as_list() == list(
        mock_env_specs.get_agent_environment_specs()["agent_1"].actions.shape
    )
    assert parallel_transition_adder_signature.name() == "data_server_adder_signature"


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
    assert uniform_priority.name() == "adder_priority"
    assert issubclass(
        uniform_priority.__init__.__annotations__["config"], SimpleNamespace  # type: ignore # noqa:E501
    )
    assert all(mock_builder.store.priority_fns) == 1
