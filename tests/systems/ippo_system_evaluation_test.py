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

from mava.systems.system import System
from tests.systems.systems_test_data import ippo_system_multi_thread_eval

#########################################################################
# Full system integration test with evaluation interval.


@pytest.fixture
def test_ippo_system_mt() -> System:
    """A multi threaded built system that uses Launchpad"""
    return ippo_system_multi_thread_eval()


# TODO Re-add once issue is solved
# https://github.com/instadeepai/Mava/issues/842
@pytest.mark.skip
def test_ippo(
    test_ippo_system_mt: System,
) -> None:
    """Full integration test of ippo system."""

    test_ippo_system_mt.launch()
