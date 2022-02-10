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
from typing import Any, List
from types import SimpleNamespace

import pytest

from mava.core_jax import BaseSystem


CONFIG = SimpleNamespace()


@dataclass
class parameter_set_1:
    str_param: str


@dataclass
class parameter_set_2:
    int_param: int
    float_param: float


CONFIG.set_1 = parameter_set_1(str_param="param")
CONFIG.set_2 = parameter_set_2(int_param=4, float_param=5.4)


class IncompleteSystem(BaseSystem):
    def build(self, config) -> SimpleNamespace:
        assert config.set_1.str_param == "param"
        assert config.set_2.int_param == 4
        assert config.set_2.float_param == 5.4


class CompleteSystem(IncompleteSystem):
    def update(self, component: Any, name: str) -> None:
        pass

    def add(self, component: Any, name: str) -> None:
        pass

    def launch(
        self,
        num_executors: int,
        multi_process: str,
        nodes_on_gpu: List[str],
        name: str,
    ):
        pass


@pytest.fixture
def base_system_complete():
    return CompleteSystem(CONFIG)


# def test_core_base_system():
