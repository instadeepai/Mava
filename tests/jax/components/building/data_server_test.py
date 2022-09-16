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

"""Tests for data server components of Jax-based Mava systems"""

from types import SimpleNamespace
from typing import Any, List

import pytest
import reverb

from mava.adders import reverb as reverb_adders
from mava.callbacks.base import Callback
from mava.components.building.data_server import OffPolicyDataServer, OnPolicyDataServer
from mava.components.building.environments import EnvironmentSpec, EnvironmentSpecConfig
from mava.systems.builder import Builder
from mava.utils import enums
from tests.jax.mocks import make_fake_environment_factory


class MockBuilder(Builder):
    def __init__(
        self,
        components: List[Callback],
        global_config: SimpleNamespace = SimpleNamespace(),
    ) -> None:
        """Initialize mock builder for tests."""
        super().__init__(components, global_config)


@pytest.fixture
def mock_builder() -> Builder:
    """Creates mock builder.

    Here the system has shared weights and fixed network sampling.

    Returns:
        Builder: Mava builder
    """

    # Create environment spec for adding to builder
    environment_spec = EnvironmentSpec(
        EnvironmentSpecConfig(environment_factory=make_fake_environment_factory())
    )

    builder = MockBuilder(
        components=[],
        global_config=SimpleNamespace(
            network_sampling_setup_type=enums.NetworkSampler.fixed_agent_networks
        ),
    )

    # Add environment spec to builder
    environment_spec.on_building_init_start(builder)

    builder.store.table_network_config = {
        "trainer_0": ["network_agent", "network_agent", "network_agent"]
    }
    builder.store.agent_net_keys = {
        "agent_0": "network_agent",
        "agent_1": "network_agent",
        "agent_2": "network_agent",
    }
    builder.store.extras_spec = {}

    return builder


@pytest.mark.parametrize(
    "sampler, remover, sampler_field, remover_field",
    [
        (reverb.selectors.Uniform(), reverb.selectors.Lifo(), "uniform", "lifo"),
        (reverb.selectors.MinHeap(), reverb.selectors.Fifo(), "heap", "fifo"),
        (
            reverb.selectors.Prioritized(0.5),
            reverb.selectors.Lifo(),
            "prioritized",
            "lifo",
        ),
    ],
)
def test_off_policy_data_server(
    mock_builder: Builder,
    sampler: Any,
    remover: Any,
    sampler_field: str,
    remover_field: str,
) -> None:
    """First test for data server.

    Args:
        mock_builder : Mava builder
        sampler: reverb sampler
        remover: reverb remover
        sampler_field: string to test remover type
        remover_field: string to test remover type
    """

    mock_builder.store.rate_limiter_fn = lambda: reverb.rate_limiters.MinSize(1000)
    mock_builder.store.adder_signature_fn = (
        lambda x, y: reverb_adders.ParallelNStepTransitionAdder.signature(x, y)
    )
    mock_builder.store.sampler_fn = lambda: sampler
    mock_builder.store.remover_fn = lambda: remover

    data_server = OffPolicyDataServer()
    data_server.on_building_data_server(mock_builder)

    table = mock_builder.store.data_tables[0]

    assert table.info.name == "trainer_0"
    assert table.info.rate_limiter_info.min_size_to_sample == 1000
    assert table.info.sampler_options.HasField(sampler_field)
    assert table.info.remover_options.HasField(remover_field)
    assert table.info.max_size == 100000
    assert table.info.max_times_sampled == 0

    # check table signature type
    assert type(table.info.signature).__name__ == "Transition"


def test_on_policy_data_server_no_sequence_length(
    mock_builder: Builder,
) -> None:
    """Tests on policy data server when no sequence length is given"""

    mock_builder.store.adder_signature_fn = lambda env_specs, extras_specs: reverb_adders.ParallelNStepTransitionAdder.signature(  # noqa: E501
        env_specs, extras_specs
    )

    data_server = OnPolicyDataServer()
    data_server.on_building_data_server(mock_builder)

    table = mock_builder.store.data_tables[0]

    assert table.info.max_size == 1000
    assert table.info.name == "trainer_0"
    assert type(table.info.signature).__name__ == "Transition"


def test_on_policy_data_server_with_sequence_length(
    mock_builder: Builder,
) -> None:
    """Tests on policy data server when sequence length and \
        sequence adder is given"""

    mock_builder.store.adder_signature_fn = lambda env_specs, seq_length, extras_specs: reverb_adders.ParallelSequenceAdder.signature(  # noqa: E501
        env_specs, seq_length, extras_specs
    )

    mock_builder.store.global_config = SimpleNamespace(sequence_length=20)

    data_server = OnPolicyDataServer()
    data_server.on_building_data_server(mock_builder)

    table = mock_builder.store.data_tables[0]

    assert table.info.max_size == 1000
    assert table.info.name == "trainer_0"
    assert type(table.info.signature).__name__ == "Step"
