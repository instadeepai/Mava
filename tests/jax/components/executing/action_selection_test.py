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

"""Tests for FeedforwardExecutorSelectAction class for Jax-based Mava systems"""

from types import SimpleNamespace

import jax
import pytest
from acme.types import NestedArray

from mava.components.jax.executing.action_selection import (
    ExecutorSelectActionConfig,
    FeedforwardExecutorSelectAction,
)
from mava.systems.jax.executor import Executor


@pytest.fixture
def dummy_config() -> ExecutorSelectActionConfig:
    """Dummy config attribute for FeedforwardExecutorSelectAction class

    Returns:
        ExecutorSelectActionConfig
    """
    config = ExecutorSelectActionConfig()
    config.parm_0 = 1
    return config


@pytest.fixture
def dummy_ff_executor_select_action():
    """Dummy FeedforwardExecutorSelectAction.

    Returns:
        FeedforwardExecutorSelectAction
    """
    return FeedforwardExecutorSelectAction()


@pytest.fixture
def mock_empty_executor() -> Executor:
    """Mock executore component with empty observations"""
    store = SimpleNamespace(is_evaluator=None, observations={})
    return Executor(store=store)


def get_action(observation, rng_key, legal_actions):
    """Function used in the networks.

    Returns:
        action_info and policy info
    """
    return "action_info_after_get_action", "policy_info_after_get_action"


class MockExecutor(Executor):
    def __init__(self):
        observations = {
            "agent_0": [0.1, 0.5, 0.7],
            "agent_1": [0.8, 0.3, 0.7],
            "agent_2": [0.9, 0.9, 0.8],
        }
        agent_net_keys = {
            "agent_0": "network_agent",
            "agent_1": "network_agent",
            "agent_2": "network_agent",
        }
        networks = {
            "networks": {
                agent_net_keys["agent_0"]: SimpleNamespace(get_action=get_action),
                agent_net_keys["agent_1"]: SimpleNamespace(get_action=get_action),
                agent_net_keys["agent_2"]: SimpleNamespace(get_action=get_action),
            }
        }
        key = jax.random.PRNGKey(5)
        action_info = "action_info_test"
        policy_info = "policy_info_test"

        store = SimpleNamespace(
            is_evaluator=None,
            observations=observations,
            observation=SimpleNamespace(observation=[0.1, 0.5, 0.7], legal_actions=[1]),
            agent="agent_0",
            networks=networks,
            agent_net_keys=agent_net_keys,
            key=key,
            action_info=action_info,
            policy_info=policy_info,
        )
        self.store = store

    def select_action(
        self, agent: str, observation: NestedArray, state: NestedArray = None
    ) -> NestedArray:
        action_info = "action_info_" + str(agent)
        policy_info = "policy_info_" + str(agent)
        return action_info, policy_info


@pytest.fixture
def mock_executor() -> Executor:
    """Mock executor component."""
    return MockExecutor()


# Test initiator
def test_constructor(dummy_config: ExecutorSelectActionConfig) -> None:
    """Test adding config as an attribute

    Args:
        dummy_config
    """
    ff_executor_select_action = FeedforwardExecutorSelectAction(config=dummy_config)
    assert ff_executor_select_action.config.parm_0 == dummy_config.parm_0


# Test on_execution_select_actions
def test_on_execution_select_actions_with_empty_observations(
    mock_empty_executor: Executor,
    dummy_ff_executor_select_action: FeedforwardExecutorSelectAction,
) -> None:
    """Test on_execution_select_actions with empty observations

    Args:
        mock_empty_executor: executor with no observations and no agents
        dummy_ff_executor_select_action: FeedforwardExecutorSelectAction
    """
    dummy_ff_executor_select_action.on_execution_select_actions(
        executor=mock_empty_executor
    )

    assert mock_empty_executor.store.actions_info == {}
    assert mock_empty_executor.store.policies_info == {}


def test_on_execution_select_actions(
    mock_executor: Executor,
    dummy_ff_executor_select_action: FeedforwardExecutorSelectAction,
) -> None:
    """Test on_execution_select_actions.

    Args:
        dummy_config: config
        mock_executor: Executor
    """
    dummy_ff_executor_select_action.on_execution_select_actions(executor=mock_executor)

    for agent in mock_executor.store.observations.keys():
        assert mock_executor.store.actions_info[agent] == "action_info_" + agent
        assert mock_executor.store.policies_info[agent] == "policy_info_" + agent


# Test on_execution_select_action_compute
def test_on_execution_select_action_compute(
    mock_executor: Executor,
    dummy_ff_executor_select_action: FeedforwardExecutorSelectAction,
) -> None:
    """Test on_execution_select_action_compute.

    Args:
        dummy_config: config
        mock_executor: Executor
    """
    dummy_ff_executor_select_action.on_execution_select_action_compute(
        executor=mock_executor
    )

    assert mock_executor.store.action_info == "action_info_after_get_action"
    assert mock_executor.store.policy_info == "policy_info_after_get_action"
