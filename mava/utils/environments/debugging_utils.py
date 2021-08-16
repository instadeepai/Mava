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

from typing import Optional

import dm_env

from mava.utils.debugging.environments import TwoStepEnv, switch_game
from mava.utils.debugging.make_env import make_debugging_env
from mava.wrappers.debugging_envs import (
    DebuggingEnvWrapper,
    SwitchGameWrapper,
    TwoStepWrapper,
)


def make_environment(
    evaluation: bool = None,
    env_name: str = "simple_spread",
    action_space: str = "discrete",
    num_agents: int = 3,
    render: bool = False,
    return_state_info: bool = False,
    random_seed: Optional[int] = None,
) -> dm_env.Environment:

    assert action_space == "continuous" or action_space == "discrete"

    del evaluation

    if env_name == "two_step":
        environment = TwoStepEnv()
        environment = TwoStepWrapper(environment)
    elif env_name == "switch":
        """Creates a SwitchGame environment."""
        env_module_fn = switch_game.MultiAgentSwitchGame(num_agents=num_agents)
        environment_fn = SwitchGameWrapper(env_module_fn)
        return environment_fn
    else:
        """Creates a MPE environment."""
        env_module = make_debugging_env(env_name, action_space, num_agents, random_seed)
        environment = DebuggingEnvWrapper(
            env_module, return_state_info=return_state_info
        )

    if random_seed and hasattr(environment, "seed"):
        environment.seed(random_seed)

    return environment
