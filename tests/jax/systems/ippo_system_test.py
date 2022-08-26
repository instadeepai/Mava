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


import pytest

from mava.systems.jax.system import System
from tests.jax.systems.test_systems import test_ippo_system_multi_process

#########################################################################
# Full system integration test.


@pytest.fixture
def test_ippo_system_mp() -> System:
    """A multi process built system"""
    return test_ippo_system_multi_process()


def test_ippo(
    test_ippo_system_mp: System,
) -> None:
    """Full integration test of ippo system."""
    (trainer_node,) = test_ippo_system_mp._builder.store.program._program._groups[
        "trainer"
    ]

    trainer_node.disable_run()

    test_ippo_system_mp.launch()
    trainer_run = trainer_node.create_handle().dereference()

    for _ in range(5):
        trainer_run.step()
