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

import jaxmarl
import jumanji
from jaxmarl.environments.smax import map_name_to_scenario
from jumanji.env import Environment
from jumanji.environments.routing.lbf.generator import (
    RandomGenerator as LbfRandomGenerator,
)
from jumanji.environments.routing.robot_warehouse.generator import (
    RandomGenerator as RwareRandomGenerator,
)
from jumanji.wrappers import AutoResetWrapper

import mava.utils.jumanji_registry as jumanji_registry
from mava.wrappers.jaxmarl import JaxMarlWrapper
from mava.wrappers.jumanji import LbfWrapper, RwareWrapper
from mava.wrappers.shared import AgentIDWrapper, GlobalStateWrapper, LogWrapper

# Registry mapping environment names to their generator and wrapper classes.
_jumanji_wrappers_and_args = {
    "RobotWarehouse-v0": {"generator": RwareRandomGenerator, "wrapper": RwareWrapper},
    "LevelBasedForaging-v0": {"generator": LbfRandomGenerator, "wrapper": LbfWrapper},
}


def add_optional_wrappers(env: Environment, config: Dict) -> Environment:
    # Add the global state to observation.
    if config["system"]["add_global_state"]:
        env = GlobalStateWrapper(env)

    # Add agent id to observation.
    if config["system"]["add_agent_id"]:
        env = AgentIDWrapper(env, config["system"]["add_global_state"])

    return env


def make_jumanji_env(env_name: str, config: Dict) -> Tuple[Environment, Environment]:
    """
    Create a Jumanji environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    # Config generator and select the wrapper.
    task_config = jumanji_registry.get_task_config(
        config["env"]["env_name"], config["env"]["scenario"]
    )
    config["env"]["scenario"] = {config["env"]["scenario"]: task_config}

    generator = _jumanji_wrappers_and_args[env_name]["generator"]
    generator = generator(**task_config)
    wrapper = _jumanji_wrappers_and_args[env_name]["wrapper"]

    # Create envs.
    env = jumanji.make(env_name, generator=generator, time_limit=config["env"]["time_limit"])
    env = wrapper(env)
    eval_env = jumanji.make(env_name, generator=generator, time_limit=config["env"]["time_limit"])
    eval_env = wrapper(eval_env)

    env = add_optional_wrappers(env, config)
    eval_env = add_optional_wrappers(eval_env, config)

    env = AutoResetWrapper(env)
    env = LogWrapper(env)

    return env, eval_env


def make_jaxmarl_env(env_name: str, config: Dict) -> Tuple[Environment, Environment]:
    """
     Create a JAXMARL environment.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A JAXMARL environment.
    """

    kwargs = config["env"]["kwargs"]
    if "smax" in env_name.lower():
        kwargs["scenario"] = map_name_to_scenario(config["env"]["scenario"])

    # Create jaxmarl envs.
    env = JaxMarlWrapper(jaxmarl.make(env_name, **kwargs), config["system"]["add_global_state"])
    eval_env = JaxMarlWrapper(
        jaxmarl.make(env_name, **kwargs), config["system"]["add_global_state"]
    )

    # Add optional wrappers.
    if config["system"]["add_agent_id"]:
        env = AgentIDWrapper(env, config["system"]["add_global_state"])
        eval_env = AgentIDWrapper(eval_env, config["system"]["add_global_state"])

    env = AutoResetWrapper(env)
    env = LogWrapper(env)

    return env, eval_env


def make(config: Dict) -> Tuple[Environment, Environment]:
    """
    Create environments for training and evaluation..

    Args:
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    env_name = config["env"]["env_name"]

    if env_name in _jumanji_wrappers_and_args:
        return make_jumanji_env(env_name, config)
    elif env_name in jaxmarl.registered_envs:
        return make_jaxmarl_env(env_name, config)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")
