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

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, List

import pytest

from mava.core_jax import BaseSystem

# TODO(Arnu): update to use Mava config class for Jax systems when possible
# Dummy config and component
CONFIG = SimpleNamespace()
COMPONENT = None


@dataclass
class parameter_set_1:
    str_param: str


@dataclass
class parameter_set_2:
    int_param: int
    float_param: float


CONFIG.set_1 = parameter_set_1(str_param="param")
CONFIG.set_2 = parameter_set_2(int_param=4, float_param=5.4)


class TestDummySystem(BaseSystem):
    """Create a complete class with all abstract method overwritten."""

    def update(self, component: Any, name: str) -> None:
        """Dummy update"""
        assert component is None
        assert name == "update"

    def add(self, component: Any, name: str) -> None:
        """Dummy add"""
        assert component is None
        assert name == "add"

    def build(self, config: SimpleNamespace) -> SimpleNamespace:
        """Dummy build"""
        assert config.set_1.str_param == "param"
        assert config.set_2.int_param == 4
        assert config.set_2.float_param == 5.4

        # hypothetical system components
        components = SimpleNamespace()
        return components

    def distribute(
        self,
        num_executors: int,
        nodes_on_gpu: List[str],
        distributor: Any = None,
    ) -> None:
        """Dummy distribute"""
        assert num_executors == 1
        assert nodes_on_gpu[0] == "process"
        assert distributor is None

    def launch(self, name: str = "system") -> None:
        """Dummy launch"""
        assert name == "system"


@pytest.fixture
def dummy_system() -> TestDummySystem:
    """Create complete system for use in tests"""
    return TestDummySystem()


def test_dummy_system(dummy_system: TestDummySystem) -> None:
    """Test complete system methods"""
    # update component
    dummy_system.update(COMPONENT, "update")

    # add component
    dummy_system.add(COMPONENT, "add")

    # build system
    dummy_system.build(CONFIG)

    # distribute system
    dummy_system.distribute(num_executors=1, nodes_on_gpu=["process"])

    # launch system
    dummy_system.launch()
