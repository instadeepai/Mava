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

from mava.utils.config_utils import flatten_dict

test_mock_nested_dict = {"a": 1, "c": {"a": 2, "b": {"x": 5, "y": 10}}, "d": [1, 2, 3]}


class TestConfigUtils:
    def test_flatten_dict(self) -> None:
        """Tests flatten_dict."""
        assert flatten_dict(d=test_mock_nested_dict) == {
            "a": 1,
            "c.a": 2,
            "c.b.x": 5,
            "c.b.y": 10,
            "d": [1, 2, 3],
        }

    def test_flatten_dict_different_sep(self) -> None:
        """Tests flatten dict with specified sep token."""
        assert flatten_dict(d=test_mock_nested_dict, sep="_") == {
            "a": 1,
            "c_a": 2,
            "c_b_x": 5,
            "c_b_y": 10,
            "d": [1, 2, 3],
        }

    def test_flatten_dict_with_parent_key(self) -> None:
        """Tests flatten dict with specified parent key."""
        assert flatten_dict(d=test_mock_nested_dict, parent_key="test_parent") == {
            "test_parent.a": 1,
            "test_parent.c.a": 2,
            "test_parent.c.b.x": 5,
            "test_parent.c.b.y": 10,
            "test_parent.d": [1, 2, 3],
        }
