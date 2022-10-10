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
from typing import Any, Dict, Optional, Tuple

from mava.utils.jax_training_utils import set_jax_double_precision

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
        concat_prev_actions: bool = False,
        concat_agent_id: bool = False,
        evaluation: bool = False,
        random_seed: Optional[int] = None,
        death_masking: bool = False,
    ) -> Tuple[Any, Dict[str, str]]:
        """Make a SMAC environment wrapper.

        Args:
            map_name: the name of the scenario
            concat_prev_actions: Concat one-hot vector of agent prev_action to obs.
            concat_agent_id: Concat one-hot vector of agent ID to obs.
            evaluation: extra param for evaluation
            random_seed: seed
            death_masking: whether to mask out agent observations once dead
        """
        # Env uses int64 action space due to the use of spac.Discrete.
        set_jax_double_precision()
        env = StarCraft2Env(map_name=map_name, seed=random_seed)

        env = SMACWrapper(env, death_masking=death_masking)

        if concat_prev_actions:
            env = ConcatPrevActionToObservation(env)

        if concat_agent_id:
            env = ConcatAgentIdToObservation(env)

        environment_task_name = {"environment_name": "SMAC", "task_name": map_name}
        return env, environment_task_name
