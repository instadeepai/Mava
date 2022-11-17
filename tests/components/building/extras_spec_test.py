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

"""Extras Spec unit test"""
from types import SimpleNamespace
from typing import Dict, List
from unittest.mock import patch

import pytest
from dm_env import specs

from mava.components.building.extras_spec import ExtrasSpec


@pytest.fixture
# Allows testing of abstract class
@patch.multiple(ExtrasSpec, __abstractmethods__=set())
def mock_extras_spec() -> ExtrasSpec:
    """Mock ExtrasSpec component.

    Returns:
        A base ExtrasSpec component
    """
    return ExtrasSpec()  # type: ignore


def test___init__(mock_extras_spec: ExtrasSpec) -> None:
    """Tests that default init makes a config with a SimpleNamespace"""
    assert mock_extras_spec.config == SimpleNamespace()


def test_name(mock_extras_spec: ExtrasSpec) -> None:
    """Tests that the name of the component is extras_spec"""
    assert mock_extras_spec.name() == "extras_spec"


def test_get_network_keys_no_agents(mock_extras_spec: ExtrasSpec) -> None:
    """Tests that get_network_keys returns an empty mapping when there are no agents"""
    unique_net_keys: List[int] = [1]
    agent_ids: List[str] = []

    expected_agent_net_keys: Dict[str, Dict[str, specs.DiscreteArray]] = {
        "network_keys": {}
    }

    assert (
        mock_extras_spec.get_network_keys(unique_net_keys, agent_ids)
        == expected_agent_net_keys
    )


def test_get_network_keys_with_agents(mock_extras_spec: ExtrasSpec) -> None:
    """Tests that get_network_keys maps agents to all networks"""
    unique_net_keys: List[int] = [1, 2]
    agent_ids: List[str] = ["agent_0", "agent_1", "agent_2"]

    expected_agent_net_keys: Dict[str, Dict[str, specs.DiscreteArray]] = {
        "network_keys": {
            "agent_0": specs.DiscreteArray(2),
            "agent_1": specs.DiscreteArray(2),
            "agent_2": specs.DiscreteArray(2),
        }
    }

    assert (
        mock_extras_spec.get_network_keys(unique_net_keys, agent_ids)
        == expected_agent_net_keys
    )
