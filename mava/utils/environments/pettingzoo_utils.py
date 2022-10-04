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

"""Pettingzoo environment factory."""

import importlib
from typing import Any, List, Optional, Union

import dm_env
import numpy as np

from mava.utils.jax_training_utils import set_jax_double_precision

try:
    import supersuit

    _has_supersuit = True
except ModuleNotFoundError:
    _has_supersuit = False

try:
    from smac.env.pettingzoo import StarCraft2PZEnv

    _has_smac = True
except ModuleNotFoundError:
    _has_smac = False

try:
    import pettingzoo  # noqa: F401

    _has_petting_zoo = True
except ModuleNotFoundError:
    _has_petting_zoo = False

from mava.wrappers import (
    ParallelEnvWrapper,
    PettingZooAECEnvWrapper,
    PettingZooParallelEnvWrapper,
    SequentialEnvWrapper,
)


def atari_preprocessing(
    env: Union[ParallelEnvWrapper, SequentialEnvWrapper]
) -> Union[ParallelEnvWrapper, SequentialEnvWrapper]:
    """Preprocess atari env.

    Args:
        env : a mava supported atari env.

    Returns:
        a wrapped mava env.
    """
    if _has_supersuit:
        # Preprocessing
        env = supersuit.max_observation_v0(env, 2)

        # repeat_action_probability is set to 0.25
        # to introduce non-determinism to the system
        env = supersuit.sticky_actions_v0(env, repeat_action_probability=0.25)

        # skip frames for faster processing and less control
        # to be compatable with gym, use frame_skip(env, (2,5))
        env = supersuit.frame_skip_v0(env, 4)

        # downscale observation for faster processing
        env = supersuit.resize_v0(env, 84, 84)

        # allow agent to see everything on the screen
        # despite Atari's flickering screen problem
        env = supersuit.frame_stack_v1(env, 4)

        # set dtype to float32
        env = supersuit.dtype_v0(env, np.float32)
    else:
        raise Exception("Supersuit is not installed.")
    return env


def make_environment(
    evaluation: bool = False,
    env_class: str = "mpe",
    env_name: str = "simple_spread_v2",
    env_preprocess_wrappers: Optional[List] = None,
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
    if _has_petting_zoo:
        del evaluation
        set_jax_double_precision()

        if env_class == "smac":
            if _has_smac:
                env = StarCraft2PZEnv.parallel_env(map_name=env_name)
                # wrap parallel environment
                environment = PettingZooParallelEnvWrapper(
                    env, env_preprocess_wrappers=[], return_state_info=True
                )
            else:
                raise Exception("Smac is not installed.")
        else:
            env_module = importlib.import_module(f"pettingzoo.{env_class}.{env_name}")
            env = env_module.parallel_env(**kwargs)  # type: ignore
            if env_class == "atari" or "pong" in env_name:
                env = atari_preprocessing(env)
            # wrap parallel environment
            environment = PettingZooParallelEnvWrapper(
                env, env_preprocess_wrappers=env_preprocess_wrappers
            )

        if random_seed and hasattr(environment, "seed"):
            environment.seed(random_seed)
    else:
        raise Exception("Pettingzoo is not installed.")

    return environment
