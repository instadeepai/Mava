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

from mava import constants
from mava.components.executing.action_selection import (
    FeedforwardExecutorSelectAction,
    RecurrentExecutorSelectAction,
)
from mava.components.training.normalisation.observation_normalisation import (
    ObservationNormalisation,
)
from mava.components.training.normalisation.value_normalisation import (
    ValueNormalisation,
)
from mava.systems.executor import Executor
from mava.types import OLT, NestedArray


@dataclass
class DummyExecutorSelectActionConfig:
    """Dummy config for executor select action"""

    parm_0: int


@pytest.fixture
def dummy_config() -> DummyExecutorSelectActionConfig:
    """Dummy config attribute for Feedforward and Recurrent ExecutorSelectAction classes

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
def recurrent_executor_select_action() -> RecurrentExecutorSelectAction:
    """Create an object of the class RecurrentExecutorSelectAction.

    Returns:
        RecurrentExecutorSelectAction
    """
    return RecurrentExecutorSelectAction()


@pytest.fixture
def mock_empty_executor_ff() -> Executor:
    """Mock executore component with empty observations"""
    store = SimpleNamespace(
        is_evaluator=None,
        observations={},
        agent_net_keys={},
        select_actions_fn=select_actions_ff,
        base_key=42,
        policy_states=1234,
        global_config=SimpleNamespace(
            normalise_observations=False,
            normalise_target_values=False,
        ),
    )
    return Executor(store=store)


@pytest.fixture
def mock_empty_executor_recurrent() -> Executor:
    """Mock executore component with empty observations"""
    store = SimpleNamespace(
        is_evaluator=None,
        observations={},
        agent_net_keys={},
        select_actions_fn=select_actions_recurrent,
        base_key=42,
        policy_states=1234,
        global_config=SimpleNamespace(
            normalise_observations=False,
            normalise_target_values=False,
        ),
    )
    return Executor(store=store)


def get_action(
    observation: networks_lib.Observation,
    base_key: networks_lib.PRNGKey,
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


#######################
# Feedforward executors#
#######################
def select_actions_ff(
    observations: Dict[str, NestedArray],
    current_params: Dict[str, NestedArray],
    base_key: networks_lib.PRNGKey,
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
    return action_info, policy_info, base_key


class MockFeedForwardExecutor(Executor):
    """Mock for the feedforward executor"""

    def __init__(self) -> None:
        """Init for mock executor."""
        observations = {
            "agent_0": OLT(
                observation=jnp.array([0.1, 0.5, 0.7]), legal_actions=[1], terminal=[0]
            ),
            "agent_1": OLT(
                observation=jnp.array([0.8, 0.3, 0.7]), legal_actions=[1], terminal=[0]
            ),
            "agent_2": OLT(
                observation=jnp.array([0.9, 0.9, 0.8]), legal_actions=[1], terminal=[0]
            ),
        }
        agent_net_keys = {
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
            "agent_2": "network_agent_2",
        }
        policy_states = {
            "agent_0": 1234,
            "agent_1": 1234,
            "agent_2": 1234,
        }
        networks = {
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
        base_key = jax.random.PRNGKey(5)
        action_info = "action_info_test"
        policy_info = "policy_info_test"

        norm_params: Any = {
            constants.OBS_NORM_STATE_DICT_KEY: {},
        }
        for agent in ["agent_0", "agent_1", "agent_2"]:
            obs_shape = 3
            norm_params[constants.OBS_NORM_STATE_DICT_KEY][agent] = dict(
                mean=jnp.zeros(shape=obs_shape),
                var=jnp.ones(shape=obs_shape) * 4,
                std=jnp.ones(shape=obs_shape) * 2,
                count=jnp.array([10]),
            )

        store = SimpleNamespace(
            is_evaluator=None,
            observations=observations,
            observation=SimpleNamespace(observation=[0.1, 0.5, 0.7], legal_actions=[1]),
            policy_states=policy_states,
            networks=networks,
            agent_net_keys=agent_net_keys,
            base_key=base_key,
            action_info=action_info,
            policy_info=policy_info,
            select_actions_fn=select_actions_ff,
            executor_environment=SimpleNamespace(death_masked_agents=[]),
            norm_params=norm_params,
            global_config=SimpleNamespace(
                normalise_observations=True,
                normalise_target_values=False,
            ),
        )
        self.store = store
        self.callbacks = [ObservationNormalisation, ValueNormalisation]

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
def mock_feedforward_executor() -> Executor:
    """Mock executor component."""
    return MockFeedForwardExecutor()


# Test initiator
def test_ff_constructor(dummy_config: DummyExecutorSelectActionConfig) -> None:
    """Test adding config as an attribute.

    Args:
        dummy_config : dummy config for test.
    """

    ff_executor_select_action = FeedforwardExecutorSelectAction(config=dummy_config)  # type: ignore # noqa: E501
    assert ff_executor_select_action.config.parm_0 == dummy_config.parm_0


# Test on_execution_select_actions
def test_on_execution_select_actions_with_empty_observations_ff(
    mock_empty_executor_ff: Executor,
    ff_executor_select_action: FeedforwardExecutorSelectAction,
) -> None:
    """Test on_execution_select_actions with empty observations

    Args:
        mock_empty_executor_ff: executor with no observations and no agents
        ff_executor_select_action: FeedforwardExecutorSelectAction
    """
    ff_executor_select_action.on_execution_select_actions(
        executor=mock_empty_executor_ff
    )

    assert mock_empty_executor_ff.store.actions_info == {}
    assert mock_empty_executor_ff.store.policies_info == {}


def test_on_execution_select_actions_ff(
    mock_feedforward_executor: Executor,
    ff_executor_select_action: FeedforwardExecutorSelectAction,
) -> None:
    """Test on_execution_select_actions.

    Args:
        ff_executor_select_action: FeedforwardExecutorSelectAction
        mock_feedforward_executor: Executor
    """
    ff_executor_select_action.on_execution_select_actions(
        executor=mock_feedforward_executor
    )

    for agent in mock_feedforward_executor.store.observations.keys():
        assert (
            mock_feedforward_executor.store.actions_info[agent]
            == "action_info_" + agent
        )
        assert (
            mock_feedforward_executor.store.policies_info[agent]
            == "policy_info_" + agent
        )


#######################
# Recurrent executors  #
#######################
def select_actions_recurrent(
    observations: Dict[str, NestedArray],
    current_params: Dict[str, NestedArray],
    policy_states: Dict[str, NestedArray],
    key: networks_lib.PRNGKey,
) -> Tuple[
    Dict[str, NestedArray],
    Dict[str, NestedArray],
    Dict[str, NestedArray],
    networks_lib.PRNGKey,
]:
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
    return action_info, policy_info, policy_states, key


class MockRecurrentExecutor(Executor):  # type: ignore # noqa: E501
    """Mock for the recurrent executor"""

    def __init__(self) -> None:
        """Init for mock executor."""
        observations = {
            "agent_0": OLT(
                observation=jnp.array([0.1, 0.5, 0.7]), legal_actions=[1], terminal=[0]
            ),
            "agent_1": OLT(
                observation=jnp.array([0.8, 0.3, 0.7]), legal_actions=[1], terminal=[0]
            ),
            "agent_2": OLT(
                observation=jnp.array([0.9, 0.9, 0.8]), legal_actions=[1], terminal=[0]
            ),
        }
        agent_net_keys = {
            "agent_0": "network_agent_0",
            "agent_1": "network_agent_1",
            "agent_2": "network_agent_2",
        }
        policy_states = {
            "agent_0": "agent_0",
            "agent_1": "agent_1",
            "agent_2": "agent_2",
        }
        networks = {
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
        base_key = jax.random.PRNGKey(5)
        action_info = "action_info_test"
        policy_info = "policy_info_test"

        norm_params: Any = {
            constants.OBS_NORM_STATE_DICT_KEY: {},
        }
        for agent in ["agent_0", "agent_1", "agent_2"]:
            obs_shape = 3
            norm_params[constants.OBS_NORM_STATE_DICT_KEY][agent] = dict(
                mean=jnp.zeros(shape=obs_shape),
                var=jnp.ones(shape=obs_shape) * 4,
                std=jnp.ones(shape=obs_shape) * 2,
                count=jnp.array([10]),
            )

        store = SimpleNamespace(
            is_evaluator=None,
            observations=observations,
            observation=SimpleNamespace(observation=[0.1, 0.5, 0.7], legal_actions=[1]),
            policy_states=policy_states,
            networks=networks,
            agent_net_keys=agent_net_keys,
            base_key=base_key,
            action_info=action_info,
            policy_info=policy_info,
            norm_params=norm_params,
            executor_environment=SimpleNamespace(death_masked_agents=[]),
            select_actions_fn=select_actions_recurrent,
            global_config=SimpleNamespace(
                normalise_observations=True,
            ),
        )
        self.store = store
        self.callbacks = [ObservationNormalisation, ValueNormalisation]

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
def mock_recurrent_executor() -> Executor:  # type: ignore # noqa: E501
    """Mock executor component."""
    return MockRecurrentExecutor()  # type: ignore # noqa: E501


# Test initiator
def test_recurrent_constructor(dummy_config: DummyExecutorSelectActionConfig) -> None:  # type: ignore # noqa: E501
    """Test adding config as an attribute.

    Args:
        dummy_config : dummy config for test.
    """

    recurrent_executor_select_action = RecurrentExecutorSelectAction(config=dummy_config)  # type: ignore # noqa: E501
    assert recurrent_executor_select_action.config.parm_0 == dummy_config.parm_0


# Test on_execution_select_actions
def test_on_execution_select_actions_with_empty_observations_recurrent(  # type: ignore # noqa: E501
    mock_empty_executor_recurrent: Executor,
    recurrent_executor_select_action: RecurrentExecutorSelectAction,
) -> None:
    """Test on_execution_select_actions with empty observations

    Args:
        mock_empty_executor_recurrent: executor with no observations and no agents
        recurrent_executor_select_action: RecurrentExecutorSelectAction
    """
    recurrent_executor_select_action.on_execution_select_actions(
        executor=mock_empty_executor_recurrent
    )

    assert mock_empty_executor_recurrent.store.actions_info == {}
    assert mock_empty_executor_recurrent.store.policies_info == {}


def test_on_execution_select_actions_recurrent(  # type: ignore # noqa: E501
    mock_recurrent_executor: Executor,
    recurrent_executor_select_action: RecurrentExecutorSelectAction,
) -> None:
    """Test on_execution_select_actions.

    Args:
        recurrent_executor_select_action: RecurrentExecutorSelectAction
        mock_recurrent_executor: Executor
    """
    recurrent_executor_select_action.on_execution_select_actions(
        executor=mock_recurrent_executor
    )

    for agent in mock_recurrent_executor.store.observations.keys():
        assert (
            mock_recurrent_executor.store.actions_info[agent] == "action_info_" + agent
        )
        assert (
            mock_recurrent_executor.store.policies_info[agent] == "policy_info_" + agent
        )
        assert mock_recurrent_executor.store.policy_states[agent] == agent
