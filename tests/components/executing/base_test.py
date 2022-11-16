# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

"""Testing ExecutorInit class for system builders"""

from types import SimpleNamespace

import pytest

from mava.components.executing.base import ExecutorInit, ExecutorInitConfig
from mava.systems.builder import Builder
from mava.systems.executor import Executor


@pytest.fixture
def dummy_config() -> ExecutorInitConfig:
    """Dummy config attribute for ExecutorInit class

    Returns:
        ExecutorInitConfig
    """
    return ExecutorInitConfig(
        evaluation_interval={"test": 1}, evaluation_duration={"evaluator_episodes": 32}
    )


@pytest.fixture
def mock_builder() -> Builder:
    """Mock builder component.

    Returns:
        Builder
    """
    builder = Builder(components=[])
    # store
    store = SimpleNamespace(networks=None)
    builder.store = store
    return builder


@pytest.fixture
def mock_executor() -> Executor:
    """Mock executor component.

    Returns:
        Executor
    """
    store = SimpleNamespace(is_evaluator=None)
    executor = Executor(store=store)
    executor.store.evaluation_interval = None  # type: ignore
    return executor


def test_on_execution_init_start(
    mock_executor: Executor, dummy_config: ExecutorInitConfig
) -> None:
    """Test on_execution_init_start method from ExecutorInit

    Args:
        mock_executor: Executor
        dummy_config: ExecutorInitConfig
    """
    executor_init = ExecutorInit(config=dummy_config)
    executor_init.on_execution_init_start(executor=mock_executor)

    assert mock_executor.store.evaluation_interval is None  # type: ignore # noqa: E501


def test_on_execution_init_start_with_evaluator(
    mock_executor: Executor, dummy_config: ExecutorInitConfig
) -> None:
    """Test on_execution_init_start method from ExecutorInit

    Args:
        mock_executor: Executor
        dummy_config: ExecutorInitConfig
    """
    mock_executor.store.is_evaluator = True
    executor_init = ExecutorInit(config=dummy_config)
    executor_init.on_execution_init_start(executor=mock_executor)

    assert mock_executor.store.evaluation_interval == dummy_config.evaluation_interval  # type: ignore # noqa: E501
    assert mock_executor.store.evaluation_duration == dummy_config.evaluation_duration  # type: ignore # noqa: E501


def test_name() -> None:
    """Test name method from ExecutorInit"""
    executor_init = ExecutorInit()

    assert ExecutorInit.name() == "executor_init"
    assert executor_init.name() == "executor_init"


def test_config_class() -> None:
    """Test config_class method from ExecutorInit"""
    executor_init = ExecutorInit()

    assert ExecutorInit.__init__.__annotations__["config"] == ExecutorInitConfig  # type: ignore # noqa: E501
    assert executor_init.__init__.__annotations__["config"] == ExecutorInitConfig  # type: ignore # noqa: E501
