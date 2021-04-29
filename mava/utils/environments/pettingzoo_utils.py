# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
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

"""OpenAI Gym environment factory."""

import importlib

import dm_env
import numpy as np
import supersuit

from mava.wrappers import PettingZooParallelEnvWrapper


def make_parallel_atari_environment(
    evaluation: bool = False, env_name: str = "maze_craze_v2"
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

    env_module = importlib.import_module(f"pettingzoo.atari.{env_name}")

    # TODO (Arnu): find a way to pass kwargs when using lp_utils
    env = env_module.parallel_env(game_version="race")  # type: ignore

    env = supersuit.max_observation_v0(env, 2)

    # Preprocessing
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

    # cast to parallel environment
    environment = PettingZooParallelEnvWrapper(env)
    return environment
