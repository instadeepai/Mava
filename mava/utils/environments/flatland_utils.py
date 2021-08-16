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

from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from mava.wrappers.flatland import FlatlandEnvWrapper

try:
    from flatland.envs.rail_env import RailEnv
except ModuleNotFoundError:
    pass


def load_flatland_env(env_config: Dict[str, Any]) -> RailEnv:
    """Loads a flatland environment given a config dict. Also, the possible agents in the
    environment are set"""

    env = RailEnv(**env_config)
    env.possible_agents = env.agents[:]

    return env


def flatland_env_factory(
    evaluation: bool = False,
    env_config: Dict[str, Any] = {},
    preprocessor: Callable[
        [Any], Union[np.ndarray, Tuple[np.ndarray], Dict[str, np.ndarray]]
    ] = None,
    include_agent_info: bool = False,
    random_seed: Optional[int] = None,
) -> FlatlandEnvWrapper:
    """Loads a flatand environment and wraps it using the flatland wrapper"""

    del evaluation  # since it has same behaviour for both train and eval

    env = load_flatland_env(env_config)
    wrapped_env = FlatlandEnvWrapper(env, preprocessor, include_agent_info)

    if random_seed and hasattr(wrapped_env, "seed"):
        wrapped_env.seed(random_seed)

    return wrapped_env
