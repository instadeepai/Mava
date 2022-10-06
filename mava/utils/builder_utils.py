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
import copy
from typing import Any, Callable, Dict

from mava.utils.sort_utils import sort_str_num


def copy_node_fn(fn: Callable) -> Callable:
    """Creates a copy of a node function.

    Args:
        fn : node function.

    Returns:
        copied node function.
    """
    memo = {}
    memo[id(fn.__self__.store.program)] = fn.__self__.store.program  # type: ignore
    copied_fn = copy.deepcopy(fn, memo=memo)
    return copied_fn


def convert_specs(
    agent_net_keys: Dict[str, Any], spec: Dict[str, Any], num_networks: int
) -> Dict[str, Any]:
    """_summary_

    Args:
        agent_net_keys : _description_
        spec : _description_
        num_networks : _description_
    Returns:
        _description_
    """
    if not isinstance(spec, dict):
        return spec

    agents = sort_str_num(agent_net_keys.keys())[:num_networks]
    converted_spec: Dict[str, Any] = {}
    if agents[0] in spec.keys():
        for agent in agents:
            converted_spec[agent] = spec[agent]
    else:
        # For the extras
        for key in spec.keys():
            converted_spec[key] = convert_specs(agent_net_keys, spec[key], num_networks)
    return converted_spec
