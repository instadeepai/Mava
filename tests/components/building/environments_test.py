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

"""Unit tests for environment components"""

import functools
from types import SimpleNamespace
from typing import Any, Dict, Tuple

import numpy as np
import pytest

from mava import MAEnvironmentSpec, specs
from mava.components.building.environments import (
    EnvironmentSpec,
    EnvironmentSpecConfig,
    ExecutorEnvironmentLoop,
    ExecutorEnvironmentLoopConfig,
    ParallelExecutorEnvironmentLoop,
)
from mava.core_jax import SystemBuilder
from mava.systems import Builder
from mava.utils.environments import debugging_utils
from mava.utils.sort_utils import sort_str_num


class AbstractExecutorEnvironmentLoop(ExecutorEnvironmentLoop):
    """Implement abstract methods to allow testing of class"""

    def on_building_executor_environment_loop(self, builder: SystemBuilder) -> None:
        """Do nothing. Just implement abstract method"""
        pass


class SimpleMockExecutor:
    def __init__(self) -> None:
        """Init"""
        self.store = SimpleNamespace()


@pytest.fixture
def test_environment_spec() -> EnvironmentSpec:
    """Pytest fixture for environment spec"""
    environment_factory = functools.partial(
        debugging_utils.make_environment,
        env_name="simple_spread",
        action_space="discrete",
        num_agents=10,
    )

    config = EnvironmentSpecConfig(environment_factory=environment_factory)
    test_environment_spec = EnvironmentSpec(config=config)
    return test_environment_spec


@pytest.fixture
def test_executor_environment_loop() -> ExecutorEnvironmentLoop:
    """Pytest fixture for executor environment loop"""
    config = ExecutorEnvironmentLoopConfig(should_update=False)
    return AbstractExecutorEnvironmentLoop(config=config)


@pytest.fixture
def test_parallel_executor_environment_loop() -> ParallelExecutorEnvironmentLoop:
    """Pytest fixture for executor environment loop"""

    class ExecutorStatsWrapperClass:
        """Executor stats wrapper"""

        def __init__(self, executor_environment_loop: Any) -> None:
            """Init"""
            self.executor_environment_loop = executor_environment_loop
            self.wrapped = True

    config = ExecutorEnvironmentLoopConfig(
        should_update=False,
        executor_stats_wrapper_class=ExecutorStatsWrapperClass,  # type: ignore
    )
    return ParallelExecutorEnvironmentLoop(config=config)


@pytest.fixture
def test_builder() -> SystemBuilder:
    """Pytest fixture for system builder."""

    def environment_factory(evaluation: bool) -> Tuple[str, Dict[str, str]]:
        """Function to construct the environment"""
        return "environment_eval_" + ("true" if evaluation else "false"), {
            "environment_name": "env",
            "task_name": "task",
        }

    global_config = SimpleNamespace(environment_factory=environment_factory)
    system_builder = Builder(components=[], global_config=global_config)
    system_builder.store.executor_environment = "environment"
    system_builder.store.executor = SimpleMockExecutor()
    system_builder.store.executor_logger = "executor_logger"
    system_builder.store.is_evaluator = True

    return system_builder


class TestEnvironmentSpec:
    """Tests for EnvironmentSpec"""

    def test_init(self, test_environment_spec: EnvironmentSpec) -> None:
        """Test that class loads config properly"""

        environment, _ = test_environment_spec.config.environment_factory()
        assert environment.environment.num_agents == 10

    def test_on_building_init_start(
        self, test_environment_spec: EnvironmentSpec, test_builder: SystemBuilder
    ) -> None:
        """Test by manually calling the hook and checking the store."""
        test_environment_spec.on_building_init_start(test_builder)

        # Assert for type and extra spec
        environment_spec = test_builder.store.ma_environment_spec
        assert isinstance(environment_spec, specs.MAEnvironmentSpec)
        environment, _ = test_environment_spec.config.environment_factory()

        # Assert correct spec created
        expected_spec = MAEnvironmentSpec(environment)
        assert environment_spec._extras_specs == expected_spec._extras_specs
        assert environment_spec._keys == expected_spec._keys
        for key in environment_spec._keys:
            assert (
                environment_spec._agent_environment_specs[key].observations.observation
                == expected_spec._agent_environment_specs[key].observations.observation
            )
            assert np.array_equal(
                environment_spec._agent_environment_specs[
                    key
                ].observations.legal_actions,
                expected_spec._agent_environment_specs[key].observations.legal_actions,
            )
            assert (
                environment_spec._agent_environment_specs[key].observations.terminal
                == expected_spec._agent_environment_specs[key].observations.terminal
            )
            assert (
                environment_spec._agent_environment_specs[key].actions
                == expected_spec._agent_environment_specs[key].actions
            )
            assert (
                environment_spec._agent_environment_specs[key].rewards
                == expected_spec._agent_environment_specs[key].rewards
            )
            assert (
                environment_spec._agent_environment_specs[key].discounts
                == expected_spec._agent_environment_specs[key].discounts
            )

        # Agent list
        assert test_builder.store.agents == sort_str_num(
            test_builder.store.ma_environment_spec.get_agent_ids()
        )

        # Extras spec created
        assert test_builder.store.extras_spec == {}


class TestExecutorEnvironmentLoop:
    """Tests for abstract ExecutorEnvironmentLoop"""

    def test_init(
        self, test_executor_environment_loop: ExecutorEnvironmentLoop
    ) -> None:
        """Test that class loads config properly"""
        assert not test_executor_environment_loop.config.should_update

    def test_on_building_executor_environment(
        self,
        test_executor_environment_loop: ExecutorEnvironmentLoop,
        test_builder: SystemBuilder,
    ) -> None:
        """Test by manually calling the hook and checking the store"""
        test_executor_environment_loop.on_building_executor_environment(test_builder)
        assert test_builder.store.executor_environment == "environment_eval_true"


class TestParallelExecutorEnvironmentLoop:
    """Tests for ParallelExecutorEnvironmentLoop"""

    def test_on_building_executor_environment_loop_with_stats_wrapper(
        self,
        test_parallel_executor_environment_loop: ParallelExecutorEnvironmentLoop,
        test_builder: SystemBuilder,
    ) -> None:
        """Test by calling hook with stats wrapper class in config"""
        test_parallel_executor_environment_loop.on_building_executor_environment_loop(
            test_builder
        )

        # Ensure logger deleted after it has been loaded into the environment loop
        assert not hasattr(test_builder.store, "executor_logger")

        # Assert executor_environment_loop was wrapped
        assert test_builder.store.system_executor.wrapped

        # Check that environment loop was created correctly
        executor_environment_loop = (
            test_builder.store.system_executor.executor_environment_loop
        )
        assert executor_environment_loop._environment == "environment"
        assert type(executor_environment_loop._executor) == SimpleMockExecutor
        assert executor_environment_loop._logger == "executor_logger"
        assert (
            executor_environment_loop._should_update
            == test_parallel_executor_environment_loop.config.should_update
        )

    def test_on_building_executor_environment_loop_no_stats_wrapper(
        self,
        test_parallel_executor_environment_loop: ParallelExecutorEnvironmentLoop,
        test_builder: SystemBuilder,
    ) -> None:
        """Test by calling hook with no stats wrapper class in config"""
        test_parallel_executor_environment_loop.config.executor_stats_wrapper_class = (
            None
        )
        test_parallel_executor_environment_loop.on_building_executor_environment_loop(
            test_builder
        )

        # Ensure logger deleted after it has been loaded into the environment loop
        assert not hasattr(test_builder.store, "executor_logger")

        # Check that environment loop was created correctly
        executor_environment_loop = test_builder.store.system_executor
        assert executor_environment_loop._environment == "environment"
        assert type(executor_environment_loop._executor) == SimpleMockExecutor
        assert executor_environment_loop._logger == "executor_logger"
        assert (
            executor_environment_loop._should_update
            == test_parallel_executor_environment_loop.config.should_update
        )
