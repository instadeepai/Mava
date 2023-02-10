import jax.numpy as jnp
import pytest

from mava.systems.executor import Executor
from mava.systems.idqn.components.executing.observing import (
    DQNFeedforwardExecutorObserve,
)
from tests.components.executing.observing_test import executor_without_adder  # noqa
from tests.components.executing.observing_test import mock_executor  # noqa


@pytest.fixture
def dqn_observer() -> DQNFeedforwardExecutorObserve:
    """Fixture for DQN observer"""
    return DQNFeedforwardExecutorObserve()


def test_executoin_observe_without_adder(
    executor_without_adder: Executor,  # noqa: F811
    dqn_observer: DQNFeedforwardExecutorObserve,
) -> None:
    """Tests that nothing is done when executor has no adder"""
    dqn_observer.on_execution_observe(executor_without_adder)
    assert executor_without_adder.store.extras == {
        "agent_0": jnp.array([0]),
        "agent_1": jnp.array([1]),
        "agent_2": jnp.array([2]),
    }
    assert not hasattr(executor_without_adder.store, "network_int_keys_extras")
    assert hasattr(executor_without_adder.store, "agent_net_keys")


def test_on_execution_observe(
    dqn_observer: DQNFeedforwardExecutorObserve,
    mock_executor: Executor,  # noqa: F811
) -> None:
    """Test on_execution_observe method from DQNFeedForwardExecutorObserve"""
    dqn_observer.on_execution_observe(executor=mock_executor)

    # DQN has no policy info
    assert "policy_info" not in mock_executor.store.extras

    assert (
        mock_executor.store.extras["network_int_keys"]
        == mock_executor.store.network_int_keys_extras
    )

    # test mock_executor.store.adder.add()
    assert mock_executor.store.adder.test_adder_actions == {
        "agent_0": {"actions_info": "action_info_agent_0"},
        "agent_1": {"actions_info": "action_info_agent_1"},
        "agent_2": {"actions_info": "action_info_agent_2"},
    }
    assert (
        mock_executor.store.adder.test_next_timestep
        == mock_executor.store.next_timestep
    )
