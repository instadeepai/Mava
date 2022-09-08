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

"""Tests for executor class for Jax-based Mava systems."""
from types import SimpleNamespace
from typing import Dict, List

import dm_env
import numpy as np
import pytest

from mava.callbacks import Callback
from mava.systems.jax import Executor
from mava.types import NestedArray
from tests.jax.hook_order_tracking import HookOrderTracking


class TestExecutor(HookOrderTracking, Executor):
    def __init__(
        self,
        store: SimpleNamespace,
        components: List[Callback],
    ) -> None:
        """Initialise the parameter server."""
        self.reset_hook_list()

        super().__init__(store, components)


@pytest.fixture
def test_executor() -> Executor:
    """Dummy Executor with no components."""
    return TestExecutor(
        store=SimpleNamespace(
            store_key="expected_value",
            is_evaluator=False,
        ),
        components=[],
    )


@pytest.fixture
def dummy_time_step() -> dm_env.TimeStep:
    """Dummy TimeStep."""
    return dm_env.TimeStep(
        step_type="type1", reward=-100, discount=0.9, observation=np.array([1, 2, 3])
    )


@pytest.fixture
def dummy_extras() -> Dict[str, NestedArray]:
    """Dummy extras."""
    return {"extras_key": "extras_value"}


@pytest.fixture
def dummy_actions() -> Dict[str, NestedArray]:
    """Dummy actions."""
    return {"actions_key": "actions_value"}


def test_store_loaded(test_executor: Executor) -> None:
    """Test that store is loaded during init."""
    assert test_executor.store.store_key == "expected_value"
    assert not test_executor.store.is_evaluator
    assert test_executor._evaluator == test_executor.store.is_evaluator


def test_observe_first_store(
    test_executor: Executor,
    dummy_time_step: dm_env.TimeStep,
    dummy_extras: Dict[str, NestedArray],
) -> None:
    """Test that store is handled properly in observe_first"""
    test_executor.observe_first(timestep=dummy_time_step, extras=dummy_extras)
    assert test_executor.store.timestep == dummy_time_step
    assert test_executor.store.extras == dummy_extras


def test_observe_store(
    test_executor: Executor,
    dummy_actions: Dict[str, NestedArray],
    dummy_time_step: dm_env.TimeStep,
    dummy_extras: Dict[str, NestedArray],
) -> None:
    """Test that store is handled properly in observe"""
    test_executor.observe(
        actions=dummy_actions, next_timestep=dummy_time_step, next_extras=dummy_extras
    )
    assert test_executor.store.actions == dummy_actions
    assert test_executor.store.next_timestep == dummy_time_step
    assert test_executor.store.next_extras == dummy_extras


def test_select_action_store(
    test_executor: Executor,
) -> None:
    """Test that store is handled properly in select_action"""
    agent = "agent_0"
    observation = np.array([1, 2, 3, 4])
    state = np.array([5, 6, 7, 8])

    # Manually load info into store
    test_executor.store.action_info = "action_info"
    test_executor.store.policy_info = "policy_info"

    assert test_executor.store.agent == agent
    assert (test_executor.store.observation == observation).all()
    assert (test_executor.store.state == state).all()


def test_select_actions_store(
    test_executor: Executor,
) -> None:
    """Test that store is handled properly in select_actions"""
    observations = {
        "agent_0": np.array([1, 2, 3, 4]),
        "agent_1": np.array([5, 6, 7, 8]),
    }

    # Manually load info into store
    actions_info = {"agent_0": "actions_info_0", "agent_1": "actions_info_1"}
    policies_info = {"agent_0": "policies_info_0", "agent_1": "policies_info_1"}

    test_executor.store.actions_info = actions_info
    test_executor.store.policies_info = policies_info

    assert test_executor.select_actions(observations=observations) == (
        actions_info,
        policies_info,
    )

    assert test_executor.store.observations == observations


def test_update_store(
    test_executor: Executor,
) -> None:
    """Test that store is handled properly in update"""
    test_executor.update(wait=False)
    assert not test_executor.store._wait
    test_executor.update(wait=True)
    assert test_executor.store._wait


def test_init_hook_order(test_executor: TestExecutor) -> None:
    """Test if init hooks are called in the correct order"""
    assert test_executor.hook_list == [
        "on_execution_init_start",
        "on_execution_init",
        "on_execution_init_end",
    ]


def test_observe_first_hook_order(
    test_executor: TestExecutor,
    dummy_time_step: dm_env.TimeStep,
    dummy_extras: Dict[str, NestedArray],
) -> None:
    """Test if observe_first hooks are called in the correct order"""
    test_executor.reset_hook_list()
    test_executor.observe_first(timestep=dummy_time_step, extras=dummy_extras)
    assert test_executor.hook_list == [
        "on_execution_observe_first_start",
        "on_execution_observe_first",
        "on_execution_observe_first_end",
    ]


def test_observe_hook_order(
    test_executor: TestExecutor,
    dummy_actions: Dict[str, NestedArray],
    dummy_time_step: dm_env.TimeStep,
    dummy_extras: Dict[str, NestedArray],
) -> None:
    """Test if observe hooks are called in the correct order"""
    test_executor.reset_hook_list()
    test_executor.observe(
        actions=dummy_actions, next_timestep=dummy_time_step, next_extras=dummy_extras
    )
    assert test_executor.hook_list == [
        "on_execution_observe_start",
        "on_execution_observe",
        "on_execution_observe_end",
    ]


def test_select_action_hook_order(
    test_executor: TestExecutor,
) -> None:
    """Test if select_action hooks are called in the correct order"""
    test_executor.reset_hook_list()
    test_executor.store.action_info = None
    test_executor.store.policy_info = None
   
    assert test_executor.hook_list == [
        "on_execution_select_action_start",
        "on_execution_select_action_preprocess",
        "on_execution_select_action_compute",
        "on_execution_select_action_sample",
        "on_execution_select_action_end",
    ]


def test_select_actions_hook_order(
    test_executor: TestExecutor,
) -> None:
    """Test if select_actions hooks are called in the correct order"""
    test_executor.reset_hook_list()
    test_executor.store.actions_info = None
    test_executor.store.policies_info = None
    test_executor.select_actions(observations={})
    assert test_executor.hook_list == [
        "on_execution_select_actions_start",
        "on_execution_select_actions",
        "on_execution_select_actions_end",
    ]


def test_update_hook_order(
    test_executor: TestExecutor,
) -> None:
    """Test if update hooks are called in the correct order"""
    test_executor.reset_hook_list()
    test_executor.update()
    assert test_executor.hook_list == [
        "on_execution_update_start",
        "on_execution_update",
        "on_execution_update_end",
    ]
