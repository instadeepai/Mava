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

from typing import Any, Dict, List, TypeAlias

_ConfSingleValue: TypeAlias = bool | int | float
ConfigValue: TypeAlias = _ConfSingleValue | List[_ConfSingleValue] | Dict[str, _ConfSingleValue]


def find_replace(d: Dict[str, Any], replacements: Dict[str, ConfigValue]) -> Dict[str, ConfigValue]:
    """Recursively searches through a dictionary and replaces values for specified keys.

    Args:
        d: Dictionary to search through
        replacements: The keys and values to replace
    """

    def _find_replace_recursive(current_dict: Dict[str, ConfigValue]) -> Dict[str, ConfigValue]:
        """Helper function that recursively searches and replaces values."""
        for k, v in current_dict.items():
            if isinstance(v, dict):
                current_dict[k] = _find_replace_recursive(v)
            elif k in replacements:
                current_dict[k] = replacements[k]

        return current_dict

    return _find_replace_recursive(d)
