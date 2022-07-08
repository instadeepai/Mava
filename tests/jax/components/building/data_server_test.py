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
from typing import List

import pytest
import reverb

from mava.adders import reverb as reverb_adders
from mava.callbacks.base import Callback
from mava.components.jax.building.data_server import OffPolicyDataServer
from mava.components.jax.building.environments import (
    EnvironmentSpec,
    EnvironmentSpecConfig,
)
from mava.systems.jax.builder import Builder
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
        "trainer": ["network_agent", "network_agent", "network_agent"]
    }
    builder.store.agent_net_keys = {
        "agent_0": "network_agent",
        "agent_1": "network_agent",
        "agent_2": "network_agent",
    }
    builder.store.extras_spec = {}

    return builder


def test_data_server(
    mock_builder: Builder,
) -> None:
    """First test for data server.

    Args:
        mock_builder : Mava builder
    """

    mock_builder.store.rate_limiter_fn = lambda: reverb.rate_limiters.MinSize(1000)
    mock_builder.store.adder_signature_fn = (
        lambda x, y: reverb_adders.ParallelNStepTransitionAdder.signature(x, y)
    )

    data_server = OffPolicyDataServer()
    data_server.on_building_data_server(mock_builder)

    # _table = mock_builder.store.data_tables
