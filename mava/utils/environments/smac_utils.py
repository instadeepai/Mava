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

"""Utils for SMAC environment."""
from typing import Any, Optional

try:
    from smac.env import StarCraft2Env

    _found_smac = True
except ModuleNotFoundError:
    _found_smac = False

from mava.wrappers import SMACWrapper
from mava.wrappers.env_preprocess_wrappers import (
    ConcatAgentIdToObservation,
    ConcatPrevActionToObservation,
)

if _found_smac:

    def make_environment(
        map_name: str = "3m",
        concat_prev_actions: bool = True,
        concat_agent_id: bool = True,
        evaluation: bool = False,
        random_seed: Optional[int] = None,
    ) -> Any:
        env = StarCraft2Env(map_name=map_name, seed=random_seed)

        env = SMACWrapper(env)

        if concat_prev_actions:
            env = ConcatPrevActionToObservation(env)

        if concat_agent_id:
            env = ConcatAgentIdToObservation(env)

        return env
