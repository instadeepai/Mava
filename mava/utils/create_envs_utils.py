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

from typing import Dict, Tuple

import jumanji
from jumanji.env import Environment
from jumanji.environments.routing.lbf.generator import (
    RandomGenerator as LbfRandomGenerator,
)
from jumanji.environments.routing.robot_warehouse.generator import (
    RandomGenerator as RwareRandomGenerator,
)
from jumanji.wrappers import AutoResetWrapper

from mava.wrappers.jumanji import LbfWrapper, RwareWrapper
from mava.wrappers.shared import AgentIDWrapper, LogWrapper

# Registry mapping environment names to their generator and wrapper classes.
JumanjiRegistry = {
    "RobotWarehouse-v0": {"generator": RwareRandomGenerator, "wrapper": RwareWrapper},
    "LevelBasedForaging-v0": {"generator": LbfRandomGenerator, "wrapper": LbfWrapper},
}


def create_jumanji_env(env_name: str, config: Dict) -> Tuple[Environment, Environment]:
    """
    Create a Jumanji environments for training and evaluation.
    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
    Returns:
        A tuple of the environments.
    """
    # Create envs.
    generator_class = JumanjiRegistry[env_name]["generator"]
    generator = generator_class(**config["env"]["scenario"]["task_config"])
    env = jumanji.make(env_name, generator=generator)
    wrapper_class = JumanjiRegistry[env_name]["wrapper"]
    env = wrapper_class(env)

    # Add agent id to observation.
    if config["system"]["add_agent_id"]:
        env = AgentIDWrapper(env)
    env = AutoResetWrapper(env)
    env = LogWrapper(env)

    # Create the evaluation environment.
    eval_env = jumanji.make(env_name, generator=generator)
    eval_env = wrapper_class(eval_env)
    if config["system"]["add_agent_id"]:
        eval_env = AgentIDWrapper(eval_env)

    return env, eval_env


def create_jaxmarl_env(config: Dict) -> Environment:
    """
    Create a JAXMARL environment.
    Args:
        config (Dict): The configuration of the environment.
    Returns:
        A JAXMARL environment.
    """
    # Placeholder for creating JAXMARL environment.
    pass


def create_environment(config: Dict) -> Tuple[Environment, Environment]:
    """
    Create environments for training and evaluation..
    Args:
        config (Dict): The configuration of the environment.
    Returns:
        A tuple of the environments.
    """

    env_name = config["env"]["env_name"]
    if env_name in JumanjiRegistry:
        return create_jumanji_env(env_name, config)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")
