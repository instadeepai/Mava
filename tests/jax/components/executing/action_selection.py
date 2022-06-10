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

from multiprocessing import dummy
from mava.components.jax.executing.action_selection import FeedforwardExecutorSelectAction
from mava.components.jax.executing.action_selection import ExecutorSelectActionConfig
from mava.systems.jax.executor import Executor
from tests.jax import mocks


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

@pytest.fixture
def mock_executor()->Executor:
    """ Mock executore component
    """
    observations={
        "agent_0": "observation_0",
        "agent_1": "observation_1",
        "agent_3": "observation_2",
    }
    agent_net_keys={
        "agent_0": "network_agent",
        "agent_1": "network_agent",
        "agent_2": "network_agent",
        }
    networks={
       "networks":{
           agent_net_keys["agent_0"]: get_action,
           agent_net_keys["agent_1"]: get_action,
           agent_net_keys["agent_2"]: get_action
       }
    }
    key = jax.random.PRNGKey(5)
    action_info="action_info_test"
    policy_info="policy_info_test"

    store= SimpleNamespace(
        is_evaluator= None,
        observations=observations,
        observation= observations["agent_0"],
        agent="agent_0",
        networks=networks,
        agent_net_keys=agent_net_keys,
        key=key,
        action_info=action_info,
        policy_info=policy_info
    )
    return Executor(store=store)


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

