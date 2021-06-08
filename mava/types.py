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

"""Common types used throughout Mava."""

from typing import Dict, NamedTuple, Union

import numpy as np
from acme import types
from acme.utils import loggers


class OLT(NamedTuple):
    """Container for (observation, legal_actions, terminal) tuples."""

    observation: types.Nest
    legal_actions: types.Nest
    terminal: types.Nest


NestedLogger = Union[loggers.Logger, Dict[str, loggers.Logger]]

SingleAgentAction = Union[int, float, np.array]
Action = Union[SingleAgentAction, Dict[str, SingleAgentAction]]

SingleAgentReward = Union[int, float]
Reward = Union[SingleAgentReward, Dict[str, SingleAgentReward]]
Discount = Reward

Observation = Union[OLT, Dict[str, OLT]]
