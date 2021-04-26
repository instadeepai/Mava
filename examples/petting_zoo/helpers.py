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

from mava.wrappers import PettingZooParallelEnvWrapper

TASKS = {
    "debug": ["MountainCarContinuous-v0"],
    "default": [
        "HalfCheetah-v2",
        "Hopper-v2",
        "InvertedDoublePendulum-v2",
        "InvertedPendulum-v2",
        "Reacher-v2",
        "Swimmer-v2",
        "Walker2d-v2",
    ],
}


def make_environment(
    env_type: str = "sisl",
    env_name: str = "multiwalker_v6",
    evaluation: bool = False,
    **kwargs: int,
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

    env_module = importlib.import_module(f"pettingzoo.{env_type}.{env_name}")
    env = env_module.parallel_env(**kwargs)  # type: ignore
    environment = PettingZooParallelEnvWrapper(env)
    return environment
