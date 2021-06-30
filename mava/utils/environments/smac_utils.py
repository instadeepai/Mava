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

"""Starcraft 2 environment factory."""

from typing import Any, Dict, Optional

import dm_env
from smac.env import StarCraft2Env

from mava.wrappers import SMACEnvWrapper  # type:ignore


def load_smac_env(env_config: Dict[str, Any]) -> StarCraft2Env:
    """Loads a smac environment given a config dict. Also, the possible agents in the
    environment are set"""

    env = StarCraft2Env(**env_config)
    env.possible_agents = list(range(env.n_agents))

    return env


def make_environment(
    evaluation: bool = False,
    map_name: str = "3m",
    random_seed: Optional[int] = None,
    **kwargs: Any,
) -> dm_env.Environment:
    """Wraps an starcraft 2 environment.

    Args:
        map_name: str, name of micromanagement level.

    Returns:
        A starcraft 2 smac environment wrapped as a DeepMind environment.
    """
    del evaluation

    env = StarCraft2Env(map_name=map_name, seed=random_seed, **kwargs)

    # wrap starcraft 2 environment
    environment = SMACEnvWrapper(env)

    return environment
