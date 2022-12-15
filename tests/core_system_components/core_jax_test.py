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


"""Tests for core Mava interfaces for Jax systems."""

from types import SimpleNamespace
from typing import Any, List
from unittest.mock import patch

import pytest

from mava.components.building.parameter_client import (
    BaseParameterClient,
    ExecutorParameterClient,
    TrainerParameterClient,
)
from mava.components.executing.action_selection import FeedforwardExecutorSelectAction
from mava.components.executing.observing import ExecutorObserve
from mava.core_jax import (
    BaseSystem,
    SystemBuilder,
    SystemExecutor,
    SystemParameterClient,
    SystemTrainer,
)
from mava.systems.builder import Builder


class MockBuilder(Builder):
    def __init__(self) -> None:
        """Init for mock builder"""
        self.callbacks = [ExecutorParameterClient(), FeedforwardExecutorSelectAction()]


@pytest.fixture
def builder() -> MockBuilder:
    """Fixture for mock builder"""
    return MockBuilder()


def test_exception_for_incomplete_child_system_class() -> None:
    """Test if error is thrown for missing abstract class overwrites."""
    with pytest.raises(TypeError):

        class TestIncompleteDummySystem(BaseSystem):
            def update(self, component: Any) -> None:
                """Dummy update"""
                pass

            def add(self, component: Any) -> None:
                """Dummy add"""
                pass

            def configure(self, **kwargs: Any) -> None:
                """Dummy configure"""
                pass

        TestIncompleteDummySystem()  # type: ignore


def test_exception_for_incomplete_child_builder_class() -> None:
    """Test if error is thrown for missing abstract class overwrites."""
    with pytest.raises(TypeError):

        class TestIncompleteDummySystemBuilder(SystemBuilder):
            def data_server(self) -> List[Any]:
                pass

            def executor(
                self,
                executor_id: str,
                data_server_client: Any,
                parameter_server_client: Any,
            ) -> Any:
                pass

        TestIncompleteDummySystemBuilder()  # type: ignore


def test_has_component(builder: Builder) -> None:
    """Tests if the core_component.has method works"""
    # make sure builder checks for sub types by default
    assert builder.has(BaseParameterClient)
    # make sure that subtypes work as expected
    assert builder.has(ExecutorParameterClient)
    assert not builder.has(TrainerParameterClient)

    assert builder.has(FeedforwardExecutorSelectAction)
    assert not builder.has(ExecutorObserve)


# Allows testing of abstract class
@patch.multiple(SystemTrainer, __abstractmethods__=set())
def test_system_trainer__init__() -> None:
    """Test system trainer init"""
    trainer = SystemTrainer()  # type: ignore
    assert trainer.store == SimpleNamespace()


# Allows testing of abstract class
@patch.multiple(SystemBuilder, __abstractmethods__=set())
def test_system_builder__init__() -> None:
    """Test system builder init"""
    builder = SystemBuilder()  # type: ignore
    assert builder.store == SimpleNamespace()


# Allows testing of abstract class
@patch.multiple(SystemExecutor, __abstractmethods__=set())
def test_system_executor__init__() -> None:
    """Test system executor init"""
    executor = SystemExecutor()  # type: ignore
    assert executor.store == SimpleNamespace()


# Allows testing of abstract class
@patch.multiple(SystemParameterClient, __abstractmethods__=set())
def test_system_param_client__init__() -> None:
    """Test system parameter client init"""
    param_client = SystemParameterClient()  # type: ignore
    assert param_client.store == SimpleNamespace()
