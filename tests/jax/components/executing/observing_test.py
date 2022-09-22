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

"""Tests for FeedforwardExecutorObserve class for Jax-based Mava systems"""


from types import SimpleNamespace
from typing import Any, Dict

import jax.numpy as jnp
import pytest
from dm_env import StepType, TimeStep

from mava.components.jax.executing.observing import (
    FeedforwardExecutorObserve,
    RecurrentExecutorObserve,
)
from mava.systems.jax.executor import Executor
from mava.types import OLT


class MockAdder:
    def __init__(self) -> None:
        pass

    def add_first(self, timestep: TimeStep, extras: Dict[str, Any]) -> None:
        self.test_timestep = timestep
        self.test_extras = extras

    def add(
        self,
        actions: Dict[str, Any],
        next_timestep: TimeStep,
        next_extras: Dict[str, Any],
    ) -> None:
        self.test_adder_actions = actions
        self.test_next_timestep = next_timestep
        self.test_next_extras = next_extras


class MockExecutorParameterClient:
    def __init__(self) -> None:
        self.parm = False

    def get_async(self) -> None:
        self.parm = True


# Networks
agent_net_keys = {
    "agent_0": "network_agent_0",
    "agent_1": "network_agent_1",
    "agent_2": "network_agent_2",
}


class network:
    def __init__(self) -> None:
        pass

    @staticmethod
    def get_init_state() -> int:
        return 12345


networks = {
    "network_agent_0": network,
    "network_agent_1": network,
    "network_agent_2": network,
}


@pytest.fixture
def executor_without_adder() -> Executor:
    """Mock executor component without adder"""
    extras = {
        "agent_0": jnp.array([0]),
        "agent_1": jnp.array([1]),
        "agent_2": jnp.array([2]),
    }

    store = SimpleNamespace(
        is_evaluator=None,
        observations={},
        adder=None,
        extras=extras,
        agent_net_keys=agent_net_keys,
        networks=networks,
    )
    return Executor(store=store)


class MockExecutor(Executor):
    def __init__(self, *args: object) -> None:
        # network_int_keys_extras
        network_int_keys_extras = None
        # network_sampling_setup
        network_sampling_setup = [
            ["network_agent_0"],
            ["network_agent_1"],
            ["network_agent_2"],
        ]
        # net_keys_to_ids
        net_keys_to_ids = {
            "network_agent_0": 0,
            "network_agent_1": 1,
            "network_agent_2": 2,
        }
        # timestep
        timestep = TimeStep(
            step_type=StepType.FIRST,
            reward=0.0,
            discount=1.0,
            observation=OLT(
                observation=[0.1, 0.3, 0.7], legal_actions=[1], terminal=[0.0]
            ),
        )
        # extras
        extras: Dict[str, Any] = {}
        # Policy states
        policy_states = 1234
        # Adder
        adder = MockAdder()
        # actions_info
        actions_info = {
            "agent_0": "action_info_agent_0",
            "agent_1": "action_info_agent_1",
            "agent_2": "action_info_agent_2",
        }
        # policies_info
        policies_info = {
            "agent_0": "policy_info_agent_0",
            "agent_1": "policy_info_agent_1",
            "agent_2": "policy_info_agent_2",
        }
        # executor_parameter_client
        executor_parameter_client = MockExecutorParameterClient()
        # Store
        store = SimpleNamespace(
            is_evaluator=None,
            observations={},
            policy={},
            agent_net_keys=agent_net_keys,
            network_int_keys_extras=network_int_keys_extras,
            network_sampling_setup=network_sampling_setup,
            net_keys_to_ids=net_keys_to_ids,
            timestep=timestep,
            extras=extras,
            policy_states=policy_states,
            networks=networks,
            adder=adder,
            next_extras=extras,
            next_timestep=timestep,
            actions_info=actions_info,
            policies_info=policies_info,
            executor_parameter_client=executor_parameter_client,
        )
        self.store = store


@pytest.fixture
def mock_executor() -> MockExecutor:
    """Mock executor component."""
    return MockExecutor()


@pytest.fixture
def mock_executor_fixed_net() -> MockExecutor:
    """Mock executor component with network sampling setup fixed."""
    mock_executor_fixed_network = MockExecutor()
    fixed_network_sampling_setup = [
        ["network_agent_0", "network_agent_1", "network_agent_2"]
    ]
    mock_executor_fixed_network.store.network_sampling_setup = (
        fixed_network_sampling_setup
    )
    return mock_executor_fixed_network


#######################
# Feedforward executors#
#######################
@pytest.fixture
def feedforward_executor_observe() -> FeedforwardExecutorObserve:
    """FeedforwardExecutorObserve.

    Returns:
        FeedforwardExecutorObserve
    """
    return FeedforwardExecutorObserve()


def test_on_execution_observe_first_without_adder(
    feedforward_executor_observe: FeedforwardExecutorObserve,
    executor_without_adder: Executor,
) -> None:
    """Test entering executor without store.adder

    Args:
        feedforward_executor_observe: FeedForwardExecutorObserve,
        executor_without_adder: Executor
    """
    feedforward_executor_observe.on_execution_observe_first(
        executor=executor_without_adder
    )

    assert executor_without_adder.store.extras == {
        "agent_0": jnp.array([0]),
        "agent_1": jnp.array([1]),
        "agent_2": jnp.array([2]),
    }
    assert not hasattr(executor_without_adder.store, "network_int_keys_extras")
    assert hasattr(executor_without_adder.store, "agent_net_keys")


def test_on_execution_observe_first(
    feedforward_executor_observe: FeedforwardExecutorObserve,
    mock_executor: MockExecutor,
) -> None:
    """Test on_execution_observe_first method from FeedForwardExecutorObserve

    Args:
        feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor: Executor
    """
    feedforward_executor_observe.on_execution_observe_first(executor=mock_executor)

    for agent, net in mock_executor.store.agent_net_keys.items():
        assert type(agent) == str
        assert type(net) == str
        assert agent in ["agent_0", "agent_1", "agent_2"]
        assert net in ["network_agent_0", "network_agent_1", "network_agent_2"]

    for agent, value in mock_executor.store.network_int_keys_extras.items():
        assert type(agent) == str
        assert agent in ["agent_0", "agent_1", "agent_2"]
        assert value in mock_executor.store.net_keys_to_ids.values()

    assert (
        mock_executor.store.extras["network_int_keys"]
        == mock_executor.store.network_int_keys_extras
    )

    assert mock_executor.store.adder.test_timestep == mock_executor.store.timestep
    assert mock_executor.store.adder.test_extras == mock_executor.store.extras


def test_on_execution_observe_first_fixed_sampling(
    feedforward_executor_observe: FeedforwardExecutorObserve,
    mock_executor_fixed_net: MockExecutor,
) -> None:
    """Test on_execution_observe_first method from FeedForwardExecutorObserve

    Args:
        feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor_fixed_net: Executor with fixed network sampling setup
    """
    feedforward_executor_observe.on_execution_observe_first(
        executor=mock_executor_fixed_net
    )

    assert mock_executor_fixed_net.store.agent_net_keys == {
        "agent_0": "network_agent_0",
        "agent_1": "network_agent_1",
        "agent_2": "network_agent_2",
    }

    assert mock_executor_fixed_net.store.network_int_keys_extras == {
        "agent_0": 0,
        "agent_1": 1,
        "agent_2": 2,
    }

    assert (
        mock_executor_fixed_net.store.extras["network_int_keys"]
        == mock_executor_fixed_net.store.network_int_keys_extras
    )

    # test mock_executor_fixed_net.store.adder.add_first()
    assert (
        mock_executor_fixed_net.store.adder.test_timestep
        == mock_executor_fixed_net.store.timestep
    )
    assert (
        mock_executor_fixed_net.store.adder.test_extras
        == mock_executor_fixed_net.store.extras
    )


def test_on_execution_observe_without_adder(
    feedforward_executor_observe: FeedforwardExecutorObserve,
    executor_without_adder: Executor,
) -> None:
    """Test entering executor without store.adder

    Args:
        feedforward_executor_observe: FeedForwardExecutorObserve,
        executor_without_adder: Executor
    """
    feedforward_executor_observe.on_execution_observe(executor=executor_without_adder)

    assert not hasattr(executor_without_adder.store, "next_extras")
    assert not hasattr(executor_without_adder.store.adder, "add")


def test_on_execution_observe(
    feedforward_executor_observe: FeedforwardExecutorObserve,
    mock_executor: MockExecutor,
) -> None:
    """Test on_execution_observe method from FeedForwardExecutorObserve

    Args:
        feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor: Executor
    """
    feedforward_executor_observe.on_execution_observe(executor=mock_executor)

    for agent in mock_executor.store.policies_info.keys():
        assert mock_executor.store.next_extras["policy_info"][
            agent
        ] == "policy_info_" + str(agent)

    assert (
        mock_executor.store.next_extras["network_int_keys"]
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
    assert mock_executor.store.adder.test_next_extras == mock_executor.store.next_extras


def test_on_execution_update(
    feedforward_executor_observe: FeedforwardExecutorObserve,
    mock_executor: MockExecutor,
) -> None:
    """Test on_execution_update method from FeedForwardExecutorObserve

    Args:
        feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor: Executor
    """
    feedforward_executor_observe.on_execution_update(executor=mock_executor)

    assert mock_executor.store.executor_parameter_client.parm == True


#######################
# Recurrent executors  #
#######################
@pytest.fixture
def recurrent_executor_observe() -> RecurrentExecutorObserve:
    """RecurrentExecutorObserve.

    Returns:
        RecurrentExecutorObserve
    """
    return RecurrentExecutorObserve()


def test_on_execution_observe_first_without_adder_recurrent(
    recurrent_executor_observe: RecurrentExecutorObserve,
    executor_without_adder: Executor,
) -> None:
    """Test entering executor without store.adder

    Args:
        recurrent_executor_observe: RecurrentExecutorObserve,
        executor_without_adder: Executor
    """
    recurrent_executor_observe.on_execution_observe_first(
        executor=executor_without_adder
    )

    assert executor_without_adder.store.extras == {
        "agent_0": jnp.array([0]),
        "agent_1": jnp.array([1]),
        "agent_2": jnp.array([2]),
    }
    assert not hasattr(executor_without_adder.store, "network_int_keys_extras")
    assert hasattr(executor_without_adder.store, "agent_net_keys")


def test_on_execution_observe_first_recurrent(
    recurrent_executor_observe: RecurrentExecutorObserve,
    mock_executor: MockExecutor,
) -> None:
    """Test on_execution_observe_first method from RecurrentExecutorObserve

    Args:
        recurrent_executor_observe: RecurrentExecutorObserve,
        mock_executor: Executor
    """
    recurrent_executor_observe.on_execution_observe_first(executor=mock_executor)

    for agent, net in mock_executor.store.agent_net_keys.items():
        assert type(agent) == str
        assert type(net) == str
        assert agent in ["agent_0", "agent_1", "agent_2"]
        assert net in ["network_agent_0", "network_agent_1", "network_agent_2"]

    for agent, value in mock_executor.store.network_int_keys_extras.items():
        assert type(agent) == str
        assert agent in ["agent_0", "agent_1", "agent_2"]
        assert value in mock_executor.store.net_keys_to_ids.values()

    assert (
        mock_executor.store.extras["network_int_keys"]
        == mock_executor.store.network_int_keys_extras
    )

    assert mock_executor.store.adder.test_timestep == mock_executor.store.timestep
    assert mock_executor.store.adder.test_extras == mock_executor.store.extras


def test_on_execution_observe_first_fixed_sampling_recurrent(
    recurrent_executor_observe: RecurrentExecutorObserve,
    mock_executor_fixed_net: MockExecutor,
) -> None:
    """Test on_execution_observe_first method from RecurrentExecutorObserve

    Args:
        recurrent_executor_observe: RecurrentExecutorObserve,
        mock_executor_fixed_net: Executor with fixed network sampling setup
    """
    recurrent_executor_observe.on_execution_observe_first(
        executor=mock_executor_fixed_net
    )

    assert mock_executor_fixed_net.store.agent_net_keys == {
        "agent_0": "network_agent_0",
        "agent_1": "network_agent_1",
        "agent_2": "network_agent_2",
    }

    assert mock_executor_fixed_net.store.network_int_keys_extras == {
        "agent_0": 0,
        "agent_1": 1,
        "agent_2": 2,
    }

    assert (
        mock_executor_fixed_net.store.extras["network_int_keys"]
        == mock_executor_fixed_net.store.network_int_keys_extras
    )

    # test mock_executor_fixed_net.store.adder.add_first()
    assert (
        mock_executor_fixed_net.store.adder.test_timestep
        == mock_executor_fixed_net.store.timestep
    )
    assert (
        mock_executor_fixed_net.store.adder.test_extras
        == mock_executor_fixed_net.store.extras
    )


def test_on_execution_observe_without_adder_recurrent(
    recurrent_executor_observe: RecurrentExecutorObserve,
    executor_without_adder: Executor,
) -> None:
    """Test entering executor without store.adder

    Args:
        recurrent_executor_observe: RecurrentExecutorObserve,
        executor_without_adder: Executor
    """
    recurrent_executor_observe.on_execution_observe(executor=executor_without_adder)

    assert not hasattr(executor_without_adder.store, "next_extras")
    assert not hasattr(executor_without_adder.store.adder, "add")


def test_on_execution_observe_recurrent(
    recurrent_executor_observe: RecurrentExecutorObserve,
    mock_executor: MockExecutor,
) -> None:
    """Test on_execution_observe method from RecurrentExecutorObserve

    Args:
        recurrent_executor_observe: RecurrentExecutorObserve,
        mock_executor: Executor
    """
    recurrent_executor_observe.on_execution_observe(executor=mock_executor)

    for agent in mock_executor.store.policies_info.keys():
        assert mock_executor.store.next_extras["policy_info"][
            agent
        ] == "policy_info_" + str(agent)

    assert (
        mock_executor.store.next_extras["network_int_keys"]
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
    assert mock_executor.store.adder.test_next_extras == mock_executor.store.next_extras

    # Test that policy_states are set correctly in extras
    assert mock_executor.store.adder.test_next_extras["policy_states"] == 1234


def test_on_execution_update_recurrent(
    recurrent_executor_observe: RecurrentExecutorObserve,
    mock_executor: MockExecutor,
) -> None:
    """Test on_execution_update method from RecurrentExecutorObserve

    Args:
        recurrent_executor_observe: RecurrentExecutorObserve,
        mock_executor: Executor
    """
    recurrent_executor_observe.on_execution_update(executor=mock_executor)

    assert mock_executor.store.executor_parameter_client.parm == True
