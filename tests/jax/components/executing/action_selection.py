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

from mava.components.jax.executing.action_selection import FeedforwardExecutorSelectAction
from mava.components.jax.executing.action_selection import ExecutorSelectActionConfig
from mava.systems.jax.executor import Executor

from acme.types import NestedArray
from types import SimpleNamespace
import pytest
import jax


@pytest.fixture
def dummy_config()->ExecutorSelectActionConfig:
    """Dummy config attribute for FeedforwardExecutorSelectAction class

    Returns:
        ExecutorSelectActionConfig    
    """
    config=ExecutorSelectActionConfig()
    config.parm_0=1
    return config


@pytest.fixture
def dummy_empty_config()-> ExecutorSelectActionConfig:
    """Dummy empty config attribute for FeedforwardExecutorSelectAction class

    Returns:
        ExecutorSelectActionConfig   
    """
    return ExecutorSelectActionConfig()


@pytest.fixture
def mock_empty_executor()->Executor:
    """ Mock executore component with empty observations
    """
    store= SimpleNamespace(
        is_evaluator=None,
        observations={}
    )
    return Executor(store=store)


def get_action(parm_0,parm_1,parm_2):
    """Function used in the networks.
        
    Returns:
        action_info and policy info
    """
    return "action_info_after_get_action", "policy_info_after_get_action"

class MockExecutor(Executor):
    def __init__(self):
        observations={
            "agent_0": "observation_0",
            "agent_1": "observation_1",
            "agent_2": "observation_2",
        }
        agent_net_keys={
            "agent_0": "network_agent",
            "agent_1": "network_agent",
            "agent_2": "network_agent",
            }
        networks={
        "networks":{
            agent_net_keys["agent_0"]: SimpleNamespace(get_action=get_action),
            agent_net_keys["agent_1"]: SimpleNamespace(get_action=get_action),
            agent_net_keys["agent_2"]: SimpleNamespace(get_action=get_action)
        }
        }
        key = jax.random.PRNGKey(5)
        action_info="action_info_test"
        policy_info="policy_info_test"

        store= SimpleNamespace(
            is_evaluator= None,
            observations=observations,
            observation=SimpleNamespace(observation=[0,0,1,0],legal_actions=3),
            agent="agent_0",
            networks=networks,
            agent_net_keys=agent_net_keys,
            key=key,
            action_info=action_info,
            policy_info=policy_info
        )
        self.store=store

    def select_action(self, agent: str, observation: NestedArray, state: NestedArray = None) -> NestedArray:
        action_info="action_info_"+str(agent)+"_"+str(observation)
        policy_info="policy_info_"+str(agent)+"_"+str(observation)
        return action_info, policy_info


@pytest.fixture
def mock_executor()->Executor:
    """ Mock executore component.
    """
    return MockExecutor()


# Test initiator
def test_constructor(dummy_config: type)->None:
    """Test adding config as an attribute

    Args:
        dummy_config
    """
    ff_executor_select_action=FeedforwardExecutorSelectAction(config=dummy_config)
    assert ff_executor_select_action.config.parm_0==dummy_config.parm_0


def test_constructor_empty_config(dummy_empty_config: type)->None:
    """Test adding empty config as an attribute

    Args:
        dummy_empty_config: ExecutorSelectActionConfig()
    """
    ff_executor_select_action=FeedforwardExecutorSelectAction(config=dummy_empty_config)
    assert ff_executor_select_action.config.__dict__==ExecutorSelectActionConfig().__dict__


# Test on_execution_select_actions
def test_on_execution_select_actions_with_empty_observations(mock_empty_executor:Executor, dummy_config:type)->None:
    """Test on_execution_select_actions with empty observations

    Args:
        mock_empty_executor: executor with no observations and no agents
    """
    ff_executor_select_action=FeedforwardExecutorSelectAction(dummy_config)
    ff_executor_select_action.on_execution_select_actions(executor=mock_empty_executor)

    assert mock_empty_executor.store.actions_info =={}
    assert mock_empty_executor.store.policies_info =={}


def test_on_execution_select_actions(mock_executor:Executor, dummy_config:type)->None:
    """Test on_execution_select_actions.

    Args:
        dummy_config: config
        mock_executor: Executor
    """

    ff_executor_select_action=FeedforwardExecutorSelectAction(dummy_config)
    ff_executor_select_action.on_execution_select_actions(executor=mock_executor)

    assert mock_executor.store.actions_info["agent_0"]=="action_info_agent_0_observation_0"
    assert mock_executor.store.actions_info["agent_1"]=="action_info_agent_1_observation_1"
    assert mock_executor.store.actions_info["agent_2"]=="action_info_agent_2_observation_2"

    assert mock_executor.store.policies_info["agent_0"]=="policy_info_agent_0_observation_0"
    assert mock_executor.store.policies_info["agent_1"]=="policy_info_agent_1_observation_1"
    assert mock_executor.store.policies_info["agent_2"]=="policy_info_agent_2_observation_2"


def test_on_execution_select_actions_param(dummy_config:type)->None:
    """ test errors of parameter enter in on_execution_select_actions

    Args:
        dummy_config: config
    """
    ff_executor_select_action=FeedforwardExecutorSelectAction(dummy_config)
    with pytest.raises(Exception):
        ff_executor_select_action.on_execution_select_actions(executor="Test_str_type")


#Test on_execution_select_action_compute 
def test_on_execution_select_action_compute(mock_executor:Executor, dummy_config:type)->None:
    """ Test on_execution_select_action_compute.

    Args:
        dummy_config: config
        mock_executor: Executor
    """
    ff_executor=FeedforwardExecutorSelectAction(dummy_config)
    ff_executor.on_execution_select_action_compute(executor=mock_executor)
    assert mock_executor.store.action_info=="action_info_after_get_action"
    assert mock_executor.store.policy_info=="policy_info_after_get_action"


def test_on_execution_select_action_compute_param(dummy_config:type)->None:
    """ test errors of parameter enter in on_execution_select_action_compute
    Args:
        dummy_config: config
    """
    ff_executor_select_action=FeedforwardExecutorSelectAction(dummy_config)
    with pytest.raises(Exception):
        ff_executor_select_action.on_execution_select_action_compute(executor="Test_str_type")