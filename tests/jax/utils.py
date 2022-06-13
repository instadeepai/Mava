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

"""Tests utils for Jax-based Mava systems"""
import hashlib
from typing import Any, List


def assert_if_value_is_not_none(value1: Any, value2: Any) -> None:
    """Func that compares value1 and value2, if value1 isn't none.

    Args:
        value1 : value1 to compare.
        value2 : value2 to compare.
    """
    if value1:
        assert value1 == value2


# Functions to help with asserting that hooks are called in a specific order
initial_token_value = "initial_token_value"


def hash_token(token: str, hash_by: str) -> str:
    """Use 'hash_by' to hash the given string token"""
    return hashlib.md5((token + hash_by).encode()).hexdigest()


def get_final_token_value(method_names: List[str]) -> str:
    """Get the final expected value of a token after it is hashed by the method names"""
    token = initial_token_value
    for method_name in method_names:
        token = hash_token(token, method_name)
    return token
