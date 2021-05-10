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

import dm_env

from mava.utils.debugging.make_env import make_debugging_env
from mava.wrappers.debugging_envs import DebuggingEnvWrapper


def make_environment(
    evaluation: bool,
    env_name: str = "simple_spread",
    action_space: str = "discrete",
    num_agents: int = 3,
    render: bool = False,
) -> dm_env.Environment:

    assert action_space == "continuous" or action_space == "discrete"

    del evaluation

    """Creates a MPE environment."""
    env_module = make_debugging_env(env_name, action_space, num_agents)
    environment = DebuggingEnvWrapper(env_module, render=render)
    return environment
