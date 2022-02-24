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

"""Utility functions for Mava config"""

from collections.abc import MutableMapping
from typing import List


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> MutableMapping:
    """Flattens nested dictionary structure.

    Args:
        d : dictionary to be flattened
        parent_key : optional parent key identifier
        sep : optional separator between parent and leaf keys
    Returns:
        Flattened dictionary
    """
    items: List = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)
