# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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

import importlib
from typing import Any, List, Optional, Union

import dm_env
import numpy as np

from smac.env import StarCraft2Env
from mava.wrappers import ParallelEnvWrapper, SMACEnvWrapper


def make_environment(
    evaluation: bool = False,
    map_name="3m",
    random_seed: Optional[int] = None,
    **kwargs: Any,
) -> dm_env.Environment:
    """Wraps an Pettingzoo environment.

    Args:
        env_class: str, class of the environment, e.g. MPE or Atari.
        env_name: str, name of environment, .e.g simple_spread or Pong.
        evaluation: bool, to change the behaviour during evaluation.

    Returns:
        A Pettingzoo environment wrapped as a DeepMind environment.
    """
    del evaluation

    env = StarCraft2Env(map_name=map_name, **kwargs)

    # wrap starcraft 2 environment
    environment = SMACEnvWrapper(env)

    if random_seed and hasattr(environment, "seed"):
        environment.seed(random_seed)

    return environment