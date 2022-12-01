from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import pytest
from acme.jax import networks as networks_lib

from mava.systems import Executor
from mava.systems.idqn.components.executing.action_selection import (
    DQNFeedforwardExecutorSelectAction,
)
from mava.utils.schedulers.linear_epsilon_scheduler import LinearEpsilonScheduler
from tests.components.executing.action_selection_test import (  # noqa; noqa
    mock_empty_executor,
    mock_feedforward_executor,
)


@pytest.fixture
def dqn_action_selector() -> DQNFeedforwardExecutorSelectAction:
    """Fixture for DQN action selector"""
    return DQNFeedforwardExecutorSelectAction()


def mock_select_actions(
    observations: Dict[str, jnp.ndarray],
    current_params: Dict[str, jnp.ndarray],
    base_key: networks_lib.PRNGKey,
    epsilon: float,
) -> Tuple[Dict[str, jnp.ndarray], jax.random.KeyArray]:
    """Mock action selection across all agents.

    Returns:
        a string representing agent action selection
    """
    action_info = {}
    for agent in observations.keys():
        action_info[agent] = f"action_info_{agent}_{epsilon}"
    return action_info, base_key


@pytest.fixture
def mock_empty_dqn_executor(mock_empty_executor: Executor) -> Executor:  # noqa: F811
    """Fixture to create an empty dqn executor"""
    epsilon_scheduler = LinearEpsilonScheduler(1.0, 0.0, 10)

    mock_empty_executor.store.action_selection_step = 5
    mock_empty_executor.store.episode_metrics = {}
    mock_empty_executor.store.epsilon_scheduler = epsilon_scheduler
    mock_empty_executor.store.select_actions_fn = mock_select_actions

    return mock_empty_executor


@pytest.fixture
def mock_dqn_executor(mock_feedforward_executor: Executor) -> Executor:  # noqa: F811
    """Fixture to create a dqn executor"""
    epsilon_scheduler = LinearEpsilonScheduler(1.0, 0.0, 10)

    mock_feedforward_executor.store.action_selection_step = 5
    mock_feedforward_executor.store.episode_metrics = {}
    mock_feedforward_executor.store.epsilon_scheduler = epsilon_scheduler
    mock_feedforward_executor.store.select_actions_fn = mock_select_actions

    return mock_feedforward_executor


def test_on_execution_select_actions_with_empty_observations(
    mock_empty_dqn_executor: Executor,
    dqn_action_selector: DQNFeedforwardExecutorSelectAction,
) -> None:
    """Test on_execution_select_actions with empty observations."""
    dqn_action_selector.on_execution_select_actions(executor=mock_empty_dqn_executor)

    assert mock_empty_dqn_executor.store.actions_info == {}


def test_on_execution_select_actions(
    mock_dqn_executor: Executor,
    dqn_action_selector: DQNFeedforwardExecutorSelectAction,
) -> None:
    """Test on_execution_select_actions puts the correct values in the store."""
    dqn_action_selector.on_execution_select_actions(executor=mock_dqn_executor)

    for agent in mock_dqn_executor.store.observations.keys():
        action_info = mock_dqn_executor.store.actions_info[agent]
        assert action_info == f"action_info_{agent}_0.4"

    assert mock_dqn_executor.store.episode_metrics["epsilon"] == 0.4
    assert mock_dqn_executor.store.action_selection_step == 6


def test_on_execution_select_actions_evaluator(
    mock_dqn_executor: Executor,
    dqn_action_selector: DQNFeedforwardExecutorSelectAction,
) -> None:
    """Test on_execution_select_actions when executor is an evaluator."""
    mock_dqn_executor.store.is_evaluator = True
    dqn_action_selector.on_execution_select_actions(executor=mock_dqn_executor)

    for agent in mock_dqn_executor.store.observations.keys():
        action_info = mock_dqn_executor.store.actions_info[agent]
        assert action_info == f"action_info_{agent}_0.0"

    assert mock_dqn_executor.store.episode_metrics["epsilon"] == 0.0
    assert mock_dqn_executor.store.action_selection_step == 6
