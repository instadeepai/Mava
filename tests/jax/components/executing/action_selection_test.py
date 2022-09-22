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

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import chex
import jax
import jax.numpy as jnp
import pytest
from acme.jax import networks as networks_lib
from acme.jax import utils
from acme.types import NestedArray

from mava.components.executing.action_selection import FeedforwardExecutorSelectAction
from mava.systems.executor import Executor


@dataclass
class DummyExecutorSelectActionConfig:
    """Dummy config for executor select action"""

    parm_0: int


@pytest.fixture
def dummy_config() -> DummyExecutorSelectActionConfig:
    """Dummy config attribute for FeedforwardExecutorSelectAction class

    Returns:
        ExecutorSelectActionConfig
    """
    config = DummyExecutorSelectActionConfig(parm_0=1)
    return config


@pytest.fixture
def ff_executor_select_action() -> FeedforwardExecutorSelectAction:
    """Create an object of the class FeedforwardExecutorSelectAction.

    Returns:
        FeedforwardExecutorSelectAction
    """
    return FeedforwardExecutorSelectAction()


@pytest.fixture
def mock_empty_executor() -> Executor:
    """Mock executore component with empty observations"""
    store = SimpleNamespace(
        is_evaluator=None,
        observations={},
        agent_net_keys={},
        select_actions_fn=select_actions,
        base_key=42,
    )
    return Executor(store=store)


def get_action(
    observation: networks_lib.Observation,
    rng_key: networks_lib.PRNGKey,
    legal_actions: chex.Array,
) -> Any:
    """Function used in the networks.

    Returns:
        action_info and policy info
    """
    return "action_info_after_get_action_" + str(
        observation[0]
    ), "policy_info_after_get_action_" + str(observation[0])


def get_params() -> Dict[str, jnp.ndarray]:
    """Returns dummy params for test.

    Returns:
        dummy params
    """
    return {"params": jnp.zeros(10)}


def select_actions(
    observations: Dict[str, NestedArray],
    current_params: Dict[str, NestedArray],
    key: networks_lib.PRNGKey,
) -> Tuple[Dict[str, NestedArray], Dict[str, NestedArray], networks_lib.PRNGKey]:
    """Dummy select actions.

    Args:
        observations : dummy obs.
        params : unused params.
        key : dummy key.

    Returns:
        _description_
    """
    action_info = {}
    policy_info = {}
    for agent in observations.keys():
        action_info[agent] = "action_info_" + str(agent)
        policy_info[agent] = "policy_info_" + str(agent)
    return action_info, policy_info, key


class MockExecutor(Executor):
    """Mock for the executor"""

    def __init__(self) -> None:
        """Init for mock executor."""
        observations = {
            "agent_0": [0.1, 0.5, 0.7],
            "agent_1": [0.8, 0.3, 0.7],
            "agent_2": [0.9, 0.9, 0.8],
        }
        agent_net_keys = {
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
            "agent_2": "network_agent_2",
        }
        networks = {
            "networks": {
                agent_net_keys["agent_0"]: SimpleNamespace(
                    get_action=get_action, get_params=get_params
                ),
                agent_net_keys["agent_1"]: SimpleNamespace(
                    get_action=get_action, get_params=get_params
                ),
                agent_net_keys["agent_2"]: SimpleNamespace(
                    get_action=get_action, get_params=get_params
                ),
            }
        }
        base_key = jax.random.PRNGKey(5)
        action_info = "action_info_test"
        policy_info = "policy_info_test"

        store = SimpleNamespace(
            is_evaluator=None,
            observations=observations,
            observation=SimpleNamespace(observation=[0.1, 0.5, 0.7], legal_actions=[1]),
            agent="agent_0",
            networks=networks,
            agent_net_keys=agent_net_keys,
            base_key=base_key,
            action_info=action_info,
            policy_info=policy_info,
            select_actions_fn=select_actions,
        )
        self.store = store

    def set_agent(self, agent: str) -> None:
        """Update agent, observation

        Args:
            agent: the new agent to be in store.agent
        """
        if agent not in self.store.observations.keys():
            pass

        self.store.agent = agent
        self.store.observation.observation = self.store.observations[agent]


@pytest.fixture
def mock_executor() -> Executor:
    """Mock executor component."""
    return MockExecutor()


# Test initiator
def test_constructor(dummy_config: DummyExecutorSelectActionConfig) -> None:
    """Test adding config as an attribute.

    Args:
        dummy_config : dummy config for test.
    """

    ff_executor_select_action = FeedforwardExecutorSelectAction(config=dummy_config)  # type: ignore # noqa: E501
    assert ff_executor_select_action.config.parm_0 == dummy_config.parm_0


# Test on_execution_select_actions
def test_on_execution_select_actions_with_empty_observations(
    mock_empty_executor: Executor,
    ff_executor_select_action: FeedforwardExecutorSelectAction,
) -> None:
    """Test on_execution_select_actions with empty observations

    Args:
        mock_empty_executor: executor with no observations and no agents
        ff_executor_select_action: FeedforwardExecutorSelectAction
    """
    ff_executor_select_action.on_execution_select_actions(executor=mock_empty_executor)

    assert mock_empty_executor.store.actions_info == {}
    assert mock_empty_executor.store.policies_info == {}


def test_on_execution_select_actions(
    mock_executor: Executor,
    ff_executor_select_action: FeedforwardExecutorSelectAction,
) -> None:
    """Test on_execution_select_actions.

    Args:
        ff_executor_select_action: FeedforwardExecutorSelectAction
        mock_executor: Executor
    """
    ff_executor_select_action.on_execution_select_actions(executor=mock_executor)

    for agent in mock_executor.store.observations.keys():
        assert mock_executor.store.actions_info[agent] == "action_info_" + agent
        assert mock_executor.store.policies_info[agent] == "policy_info_" + agent


# Test on_execution_select_action_compute
def test_on_execution_select_action_compute(
    mock_executor: Executor,
    ff_executor_select_action: FeedforwardExecutorSelectAction,
) -> None:
    """Test on_execution_select_action_compute.

    Args:
        ff_executor_select_action: FeedforwardExecutorSelectAction
        mock_executor: Executor
    """
    for agent in mock_executor.store.observations.keys():
        mock_executor.set_agent(agent)  # type: ignore
        ff_executor_select_action.on_execution_select_action_compute(
            executor=mock_executor
        )
        observation = utils.add_batch_dim(mock_executor.store.observations[agent])
        assert mock_executor.store.action_info == "action_info_after_get_action_" + str(
            observation[0]
        )
        assert mock_executor.store.policy_info == "policy_info_after_get_action_" + str(
            observation[0]
        )
