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

"""Tests utils for Jax-based Mava systems"""
from typing import Any


def assert_if_value_is_not_none(value1: Any, value2: Any) -> None:
    """Func that compares value1 and value2, if value1 isn't none.

    Args:
        value1 : value1 to compare.
        value2 : value2 to compare.
    """
    if value1:
        assert value1 == value2
