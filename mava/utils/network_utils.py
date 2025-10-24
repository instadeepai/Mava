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


from typing import Dict, Iterator, Tuple, Union

from gymnasium.spaces import Discrete, MultiDiscrete, Space
from hydra.utils import get_class
from jumanji.specs import DiscreteArray, MultiDiscreteArray, Spec
from omegaconf import DictConfig
from omegaconf.omegaconf import OmegaConf

from mava.networks.gnn import GNN

_DISCRETE = "discrete"
_CONTINUOUS = "continuous"


def get_action_head(action_types: Union[Spec, Space]) -> Tuple[Dict[str, str], str]:
    """Returns the appropriate action head config based on the environment action_spec."""
    if isinstance(action_types, (DiscreteArray, MultiDiscreteArray, Discrete, MultiDiscrete)):
        return {"_target_": "mava.networks.heads.DiscreteActionHead"}, _DISCRETE

    return {"_target_": "mava.networks.heads.ContinuousActionHead"}, _CONTINUOUS


def _find_key(key: str, d: dict) -> Iterator:
    """
    Recursively searches for a key in a nested dictionary and yields its values.

    This function traverses a dictionary, including any dictionaries nested within it.
    It uses a generator (`yield`) to return each matching value as it is found.

    Args:
        key (str): The key to search for.
        d (dict): The dictionary to search within.

    Yields:
        The value associated with each found instance of the key.
    """
    for k, v in d.items():
        if k == key:  # If the current key matches the target key, yield its value.
            yield v

        if isinstance(v, dict):  # If the value is another dictionary, recurse into it.
            yield from _find_key(key, v)


def is_gnn_based(config: DictConfig) -> bool:
    """Checks if any of the networks use a GNN architecture.

    Returns:
        True any of the networks use a GNN architecture, false otherwise.
    """
    net_config = OmegaConf.to_container(config.network, resolve=True)
    return any(issubclass(get_class(net), GNN) for net in _find_key("_target_", net_config))
