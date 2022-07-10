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

"""Unit tests for parameter client components"""

from types import SimpleNamespace
from typing import List

import pytest

from mava.callbacks.base import Callback
from mava.components.jax.building.parameter_client import ExecutorParameterClient
from mava.systems.jax.builder import Builder
from mava.systems.jax.parameter_server import ParameterServer


class MockBuilder(Builder):
    def __init__(
        self,
        components: List[Callback],
        global_config: SimpleNamespace = SimpleNamespace(),
    ) -> None:
        """Initialize mock builder for tests."""
        super().__init__(components, global_config)


@pytest.fixture
def mock_builder_no_evaluator_with_parameter_client() -> Builder:
    """Create a mock builder for testing that is not an evaluator \
        and that has a parameter client."""

    builder = Builder(components=[])

    builder.store.networks = {
        "networks": {
            "network_agent_0": SimpleNamespace(params={"weights": 0, "biases": 0}),
            "network_agent_1": SimpleNamespace(params={"weights": 1, "biases": 1}),
            "network_agent_2": SimpleNamespace(params={"weights": 2, "biases": 2}),
        }
    }

    builder.store.is_evaluator = False
    builder.store.parameter_server_client = ParameterServer(
        store=SimpleNamespace(get_parameters={"key_1": 1}),
        components=[],
    )

    return builder


def test_no_evaluator_with_parameter_client(
    mock_builder_no_evaluator_with_parameter_client: Builder,
) -> None:
    """Test executor parameter client.

    Args:
        mock_builder_no_evaluator_with_parameter_client : mava builder object
    """

    mock_builder = mock_builder_no_evaluator_with_parameter_client
    exec_param_client = ExecutorParameterClient()
    exec_param_client.on_building_executor_parameter_client(builder=mock_builder)

    assert hasattr(mock_builder.store, "executor_counts")
