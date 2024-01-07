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

from typing import Callable, Tuple, Union

import gym
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
from omegaconf import DictConfig

from mava.utils.jumanji_register import JumanjiScenarioManager
from mava.wrappers.gym import AgentIDWrapper as GymAgentIDWrapper
from mava.wrappers.gym import GymWrapper
from mava.wrappers.jaxmarl import JaxMarlWrapper
from mava.wrappers.jumanji import LbfWrapper, RwareWrapper
from mava.wrappers.shared import AgentIDWrapper, GlobalStateWrapper, LogWrapper

# Registry mapping environment names to their generator and wrapper classes.
_jumanji_registry = {
    "RobotWarehouse-v0": {"generator": RwareRandomGenerator, "wrapper": RwareWrapper},
    "LevelBasedForaging-v0": {"generator": LbfRandomGenerator, "wrapper": LbfWrapper},
}


def add_optional_wrappers(env: Environment, config: DictConfig) -> Environment:
    # Add agent id to observation.
    if config.system.add_agent_id:
        env = AgentIDWrapper(env)

    # Add the global state to observation.
    if config.system.add_global_state:
        env = GlobalStateWrapper(env)

    return env


def make_jumanji_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
    Create a Jumanji environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    # Config generator and select the wrapper.
    scenario_manager = JumanjiScenarioManager(env_name, config["env"]["scenario"])
    task_attributes = scenario_manager.register_environment()
    config["env"]["scenario"] = {config["env"]["scenario"]: task_attributes}

    generator = _jumanji_registry[env_name]["generator"]
    generator = generator(**task_attributes)
    wrapper = _jumanji_registry[env_name]["wrapper"]

    # Create envs.
    env = jumanji.make(env_name, generator=generator)
    env = wrapper(env)
    eval_env = jumanji.make(env_name, generator=generator)
    eval_env = wrapper(eval_env)

    env = add_optional_wrappers(env, config)
    eval_env = add_optional_wrappers(eval_env, config)

    env = AutoResetWrapper(env)
    env = LogWrapper(env)

    return env, eval_env


def make_jaxmarl_env(env_name: str, config: DictConfig) -> Tuple[Environment, Environment]:
    """
     Create a JAXMARL environment.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.

    Returns:
        A JAXMARL environment.
    """

    kwargs = config.env.kwargs
    if "smax" in env_name.lower():
        kwargs["scenario"] = map_name_to_scenario(config.env.scenario)

    # Placeholder for creating JAXMARL environment.
    env = JaxMarlWrapper(jaxmarl.make(env_name, **kwargs))
    eval_env = JaxMarlWrapper(jaxmarl.make(env_name, **kwargs))

    env = add_optional_wrappers(env, config)
    eval_env = add_optional_wrappers(eval_env, config)

    env = AutoResetWrapper(env)
    env = LogWrapper(env)

    return env, eval_env


def _make_single_gym_env(
    map_name: str = "rware-tiny-2ag-v1",
    use_individual_rewards: bool = False,
    add_agent_id: bool = True,
) -> Callable:
    """Create a function which creates a fully configured environment."""

    def thunk() -> gym.Env:
        """Create an environment."""
        env = GymWrapper(
            env=gym.make(map_name),
            use_individual_rewards=use_individual_rewards,
        )
        if add_agent_id:
            env = GymAgentIDWrapper(env)
        return env

    return thunk


def make_gym_env(
    config: DictConfig,
) -> Callable:
    """Create a gym environment.

    Note: We support currently only rware environments.
    """

    def thunk(num_envs: int) -> gym.vector.VectorEnv:
        """Create a vectorised environment."""
        if config.arch.async_envs:
            envs = gym.vector.AsyncVectorEnv(
                [
                    _make_single_gym_env(
                        map_name=config.env.scenario,
                        use_individual_rewards=config.env.use_individual_rewards,
                        add_agent_id=config.system.add_agent_id,
                    )
                    for _ in range(num_envs)
                ]
            )
        else:
            envs = gym.vector.SyncVectorEnv(
                [
                    _make_single_gym_env(
                        map_name=config.env.scenario,
                        use_individual_rewards=config.env.use_individual_rewards,
                        add_agent_id=config.system.add_agent_id,
                    )
                    for _ in range(num_envs)
                ]
            )
        envs.num_envs = num_envs
        envs.is_vector_env = True
        return envs

    return thunk


def make(config: DictConfig) -> Union[Tuple[Environment, Environment], Callable]:
    """
    Create environments for training and evaluation..

    Args:
        config (Dict): The configuration of the environment.

    Returns:
        A tuple of the environments.
    """
    env_name = config.env.env_name

    if env_name in _jumanji_registry:
        return make_jumanji_env(env_name, config)
    elif env_name in jaxmarl.registered_envs:
        return make_jaxmarl_env(env_name, config)
    elif env_name.startswith("gym"):
        return make_gym_env(config)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")
