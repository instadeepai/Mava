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

from typing import Dict, Tuple, Union

from gymnasium.spaces import Discrete, MultiDiscrete, Space
from jumanji.specs import DiscreteArray, MultiDiscreteArray, Spec

_DISCRETE = "discrete"
_CONTINUOUS = "continuous"


def get_action_head(action_types: Union[Spec, Space]) -> Tuple[Dict[str, str], str]:
    """Returns the appropriate action head config based on the environment action_spec."""
    if isinstance(action_types, (DiscreteArray, MultiDiscreteArray, Discrete, MultiDiscrete)):
        return {"_target_": "mava.networks.heads.DiscreteActionHead"}, _DISCRETE

    return {"_target_": "mava.networks.heads.ContinuousActionHead"}, _CONTINUOUS
