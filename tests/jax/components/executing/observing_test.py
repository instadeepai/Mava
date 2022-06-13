from time import time
import pytest
from mava.components.jax.executing.observing import FeedforwardExecutorObserve, ExecutorObserveConfig
from dm_env import StepType, TimeStep
from dataclasses import dataclass
from typing import Any, Dict
from mava.adders.reverb.base import ReverbParallelAdder
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num
from mava.systems.jax.executor import Executor
import pytest
from types import SimpleNamespace
from mava.types import OLT
from acme.types import NestedArray


class MockAdder:  
    def __init__(self):
        self.parm="empty"
    
    def add_first(self, timestep: TimeStep, extras: Dict[str, NestedArray] = ...) -> None:
        self.parm="after_add_first"

    def add(self, actions: Dict[str, NestedArray], next_timestep: TimeStep, next_extras: Dict[str, NestedArray] = ...) -> None:
        self.parm = "after_add"


class MockExecutorParameterClient:
    def __init__(self):
        self.parm=False

    def get_async(self):
        self.parm=True


@pytest.fixture
def dummy_config() -> ExecutorObserveConfig:
    """Dummy config attribute for FeedforwardExecutorObserve class
    Returns:
        ExecutorObserveConfig
    """
    config = ExecutorObserveConfig()
    config.parm= 1
    return config


@pytest.fixture
def mock_executor_without_adder() -> Executor:
    """Mock executore component without adder"""
    store = SimpleNamespace(is_evaluator=None, observations={}, adder=None)
    return Executor(store=store)


@pytest.fixture
def mock_executor_without_parameter_client() -> Executor:
    """Mock executore component without parameter_client"""
    store = SimpleNamespace(is_evaluator=None, observations={}, executor_parameter_client=None)
    return Executor(store=store)


class MockExecutor(Exception):
    def __init__(self, *args: object) -> None:
        # agent_net_keys
        agent_net_keys = {
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
            "agent_2": "network_agent_2",
        }
        #network_int_keys_extras
        network_int_keys_extras=None
        #network_sampling_setup
        network_sampling_setup = [
                    [
                        agent_net_keys[key]
                        for key in sort_str_num(agent_net_keys.keys())
                    ]
                ]
        #net_keys_to_ids
        all_samples = []
        for sample in network_sampling_setup:
            all_samples.extend(sample)
        unique_net_keys = list(sort_str_num(list(set(all_samples))))
        net_keys_to_ids = {
            net_key: i for i, net_key in enumerate(unique_net_keys)
        }
        #network_int_keys_extras
        network_int_keys_extras=None
        #timestep
        timestep= TimeStep(
        step_type=StepType.FIRST,
        reward=0.0,
        discount=1.0,
        observation=OLT(observation=[0.1, 0.3, 0.7], legal_actions=[1], terminal=[0.0]),
    )
        #extras
        extras={}
        #Aadder
        adder=MockAdder()
        #actions_info
        actions_info = {
            "agent_0": "action_info_agent_0",
            "agent_1": "action_info_agent_1",
            "agent_2": "action_info_agent_2",
        }
        #policies_info
        policies_info = {
            "agent_0": "policy_info_agent_0",
            "agent_1": "policy_info_agent_1",
            "agent_2": "policy_info_agent_2",
        }
        #executor_parameter_client
        executor_parameter_client=MockExecutorParameterClient()
        #Store
        store= SimpleNamespace(
            is_evaluator=None,
            observations={},
            policy={},
            agent_net_keys=agent_net_keys,
            network_int_keys_extras=network_int_keys_extras,
            network_sampling_setup=network_sampling_setup,
            net_keys_to_ids=net_keys_to_ids,
            timestep=timestep,
            extras=extras,
            adder=adder,
            next_extras=extras,
            next_timestep=timestep,
            actions_info=actions_info,
            policies_info=policies_info,
            executor_parameter_client=executor_parameter_client
        )
        self.store=store


@pytest.fixture
def mock_executor()->MockExecutor:
    """Mock executor component."""
    return MockExecutor()


@pytest.fixture
def mock_feedforward_executor_observe()->FeedforwardExecutorObserve:
    """Mock FeedforwardExecutorObserve.

    Returns:
        FeedforwardExecutorObserve
    """
    return FeedforwardExecutorObserve()


# Test initiator
def test_constructor(dummy_config: ExecutorObserveConfig) -> None:
    """Test adding config as an attribut

    Args:
        dummy_config
    """
    ff_executor_observe = FeedforwardExecutorObserve(config=dummy_config)
    
    assert ff_executor_observe.config.parm== dummy_config.parm


#Test on_execution_observe_first
def test_on_execution_observe_first_without_adder(mock_feedforward_executor_observe: FeedforwardExecutorObserve, mock_executor_without_adder: Executor)->None:
    """Test entering executor without store.adder

    Args:
        mock_feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor_without_adder: Executor
    """
    mock_feedforward_executor_observe.on_execution_observe_first(executor= mock_executor_without_adder)

    assert not mock_executor_without_adder.store.adder


def test_on_execution_observe_first(mock_feedforward_executor_observe: FeedforwardExecutorObserve, mock_executor:MockExecutor)->None:
    """Test on_execution_observe_first method from FeedForwardExecutorObserve

    Args:
        mock_feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor: Executor
    """
    mock_feedforward_executor_observe.on_execution_observe_first(executor= mock_executor)

    assert not mock_executor.store.network_int_keys_extras== None
    assert mock_executor.store.extras[
            "network_int_keys"
        ] == mock_executor.store.network_int_keys_extras
    assert mock_executor.store.adder.parm=="after_add_first"


#test on_execution_observe
def test_on_execution_observe_without_adder(mock_feedforward_executor_observe: FeedforwardExecutorObserve, mock_executor_without_adder: Executor)->None:
    """Test entering executor without store.adder

    Args:
        mock_feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor_without_adder: Executor
    """
    mock_feedforward_executor_observe.on_execution_observe(executor= mock_executor_without_adder)

    assert not mock_executor_without_adder.store.adder


def test_on_execution_observe(mock_feedforward_executor_observe: FeedforwardExecutorObserve, mock_executor: MockExecutor)->None:
    """Test on_execution_observe method from FeedForwardExecutorObserve

    Args:
        mock_feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor: Executor
    """
    mock_feedforward_executor_observe.on_execution_observe(executor= mock_executor)

    for agent in mock_executor.store.policies_info.keys():
        assert mock_executor.store.next_extras["policy_info"][agent] == "policy_info_"+str(agent)
    assert mock_executor.store.next_extras[
            "network_int_keys"
        ] == mock_executor.store.network_int_keys_extras
    assert mock_executor.store.adder.parm=="after_add"


# test on_execution_update
def test_on_execution_update_without_parameter_client(mock_feedforward_executor_observe: FeedforwardExecutorObserve, mock_executor_without_parameter_client: Executor)->None:
    """Test entering executor without store.executor_parameter_client

    Args:
        mock_feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor_without_parameter_client: Executor
    """
    mock_feedforward_executor_observe.on_execution_update(executor= mock_executor_without_parameter_client)

    assert not mock_executor_without_parameter_client.store.executor_parameter_client


def test_on_execution_update(mock_feedforward_executor_observe: FeedforwardExecutorObserve, mock_executor:MockExecutor)->None:
    """Test on_execution_update method from FeedForwardExecutorObserve

    Args:
        mock_feedforward_executor_observe: FeedForwardExecutorObserve,
        mock_executor: Executor
    """
    mock_feedforward_executor_observe.on_execution_update(executor= mock_executor)

    assert mock_executor.store.executor_parameter_client.parm==True