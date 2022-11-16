# python3
# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

from typing import Any, Dict, Optional, Tuple

import dm_env

from mava.utils.debugging.make_env import make_debugging_env
from mava.utils.jax_training_utils import set_jax_double_precision
from mava.wrappers.debugging_envs import DebuggingEnvWrapper
from mava.wrappers.env_preprocess_wrappers import ConcatAgentIdToObservation


def make_environment(
    evaluation: bool = None,
    env_name: str = "simple_spread",
    action_space: str = "discrete",
    num_agents: int = 3,
    render: bool = False,
    return_state_info: bool = False,
    random_seed: Optional[int] = None,
    recurrent_test: bool = False,
    concat_agent_id: bool = False,
) -> Tuple[dm_env.Environment, Dict[str, str]]:
    """Make a debugging environment."""

    assert action_space == "continuous" or action_space == "discrete"
    environment: Any

    if action_space == "discrete":
        # Env uses int64 action space due to the use of spac.Discrete.
        set_jax_double_precision()

    del evaluation

    if env_name == "simple_spread":
        """Creates a MPE environment."""
        env_module = make_debugging_env(
            env_name, action_space, num_agents, recurrent_test, random_seed
        )
        environment = DebuggingEnvWrapper(
            env_module, return_state_info=return_state_info
        )
    else:
        raise ValueError(f"Environment {env_name} not found.")

    if concat_agent_id:
        environment = ConcatAgentIdToObservation(environment)

    if random_seed and hasattr(environment, "seed"):
        environment.seed(random_seed)

    environment_task_name = {
        "environment_name": "debugging",
        "task_name": env_name,
    }

    return environment, environment_task_name
