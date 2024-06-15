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

from typing import Tuple

import gym
import gym.vector
import gym.wrappers
import gym.wrappers
import gym.wrappers.compatibility
import jaxmarl
import jumanji
import matrax
from gigastep import ScenarioBuilder
from jaxmarl.environments.smax import map_name_to_scenario
from jumanji.env import Environment
from jumanji.environments.routing.cleaner.generator import (
    RandomGenerator as CleanerRandomGenerator,
)
from jumanji.environments.routing.connector.generator import (
    RandomWalkGenerator as ConnectorRandomGenerator,
)
from jumanji.environments.routing.lbf.generator import (
    RandomGenerator as LbfRandomGenerator,
)
from jumanji.environments.routing.robot_warehouse.generator import (
    RandomGenerator as RwareRandomGenerator,
)
from omegaconf import DictConfig

from mava.wrappers import (
    AgentIDWrapper,
    AutoResetWrapper,
    CleanerWrapper,
    ConnectorWrapper,
    GigastepWrapper,
    GymWrapper,
    LbfWrapper,
    MabraxWrapper,
    MatraxWrapper,
    RecordEpisodeMetrics,
    RwareWrapper,
    SmaxWrapper,
)

# Registry mapping environment names to their generator and wrapper classes.
_jumanji_registry = {
    "RobotWarehouse-v0": {"generator": RwareRandomGenerator, "wrapper": RwareWrapper},
    "LevelBasedForaging-v0": {"generator": LbfRandomGenerator, "wrapper": LbfWrapper},
    "MaConnector-v2": {"generator": ConnectorRandomGenerator, "wrapper": ConnectorWrapper},
    "Cleaner-v0": {"generator": CleanerRandomGenerator, "wrapper": CleanerWrapper},
}

# Define a different registry for Matrax since it has no generator.
_matrax_registry = {"Matrax": MatraxWrapper}

_jaxmarl_wrappers = {"Smax": SmaxWrapper, "MaBrax": MabraxWrapper}

_gigastep_registry = {"Gigastep": GigastepWrapper}


def add_extra_wrappers(
    train_env: Environment, eval_env: Environment, config: DictConfig
) -> Environment:

    # Disable the AgentID wrapper if the environment has implicit agent IDs.
    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)

    # Add agent id to observation.
    if config.system.add_agent_id:
        train_env = AgentIDWrapper(train_env)
        eval_env = AgentIDWrapper(eval_env)

    train_env = AutoResetWrapper(train_env)
    train_env = RecordEpisodeMetrics(train_env)
    return train_env, eval_env


def make_jumanji_env(
    env_name: str, config: DictConfig, add_global_state: bool = False
) -> Tuple[Environment, Environment]:
    """
    Create a Jumanji environments for training and evaluation.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
        A tuple of the environments.
    """
    # Config generator and select the wrapper.
    generator = _jumanji_registry[env_name]["generator"]
    generator = generator(**config.env.scenario.task_config)
    wrapper = _jumanji_registry[env_name]["wrapper"]

    # Create envs.
    env_config = {**config.env.kwargs, **config.env.scenario.env_kwargs}
    train_env = jumanji.make(env_name, generator=generator, **env_config)
    eval_env = jumanji.make(env_name, generator=generator, **env_config)
    train_env = wrapper(train_env, add_global_state=add_global_state)
    eval_env = wrapper(eval_env, add_global_state=add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)
    return train_env, eval_env


def make_jaxmarl_env(
    env_name: str, config: DictConfig, add_global_state: bool = False
) -> Tuple[Environment, Environment]:
    """
     Create a JAXMARL environment.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
        A JAXMARL environment.
    """

    kwargs = dict(config.env.kwargs)
    if "smax" in env_name.lower():
        kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)

    # Create jaxmarl envs.
    train_env = _jaxmarl_wrappers[config.env.env_name](
        jaxmarl.make(env_name, **kwargs),
        add_global_state,
    )
    eval_env = _jaxmarl_wrappers[config.env.env_name](
        jaxmarl.make(env_name, **kwargs),
        add_global_state,
    )

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)

    return train_env, eval_env


def make_matrax_env(
    env_name: str, config: DictConfig, add_global_state: bool = False
) -> Tuple[Environment, Environment]:
    """
    Creates Matrax environments for training and evaluation.

    Args:
        env_name: The name of the environment to create.
        config: The configuration of the environment.
        add_global_state: Whether to add the global state to the observation.

    Returns:
        A tuple containing a train and evaluation Matrax environment.
    """
    # Select the Matrax wrapper.
    wrapper = _matrax_registry[env_name]

    # Create envs.
    task_name = config["env"]["scenario"]["task_name"]
    train_env = matrax.make(task_name, **config.env.kwargs)
    eval_env = matrax.make(task_name, **config.env.kwargs)
    train_env = wrapper(train_env, add_global_state)
    eval_env = wrapper(eval_env, add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)
    return train_env, eval_env


def make_gigastep_env(
    env_name: str, config: DictConfig, add_global_state: bool = False
) -> Tuple[Environment, Environment]:
    """
     Create a Gigastep environment.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation. Default False.

    Returns:
        A tuple of the environments.
    """
    wrapper = _gigastep_registry[env_name]

    kwargs = config.env.kwargs
    scenario = ScenarioBuilder.from_config(config.env.scenario.task_config)

    train_env = wrapper(scenario.make(**kwargs), has_global_state=add_global_state)
    eval_env = wrapper(scenario.make(**kwargs), has_global_state=add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)
    return train_env, eval_env


def make_gym_env(
    config: DictConfig, add_global_state: bool = False, eval_env: bool = False
) -> Environment:  # todo : create the appropriate annotation for the sync vector
    """
     Create a Gym environment.

    Args:
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation. Default False.

    Returns:
        A tuple of the environments.
    """

    def create_gym_env(
        config: DictConfig, add_global_state: bool = False, eval_env: bool = False
    ) -> Environment:  # todo: add the RecordEpisodeMetrics for gym.
        env = gym.make(config.env.scenario)
        env = gym.wrappers.compatibility.EnvCompatibility(
            env
        )  
        wrapped_env = GymWrapper(env, config.env.use_individual_rewards, add_global_state, eval_env)
        if not config.env.implicit_agent_id:
            pass  # todo : add agent id wrapper for gym .
        return wrapped_env

    num_env = config.arch.num_envs
    envs = gym.vector.AsyncVectorEnv(
        [
            lambda: create_gym_env(config, add_global_state, eval_env=eval_env)
            for _ in range(num_env)
        ]
    )

    return envs


def make(config: DictConfig, add_global_state: bool = False) -> Tuple[Environment, Environment]:
    """
    Create environments for training and evaluation..

    Args:
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
        A tuple of the environments.
    """
    env_name = config.env.scenario.name

    if env_name in _jumanji_registry:
        return make_jumanji_env(env_name, config, add_global_state)
    elif env_name in jaxmarl.registered_envs:
        return make_jaxmarl_env(env_name, config, add_global_state)
    elif env_name in _matrax_registry:
        return make_matrax_env(env_name, config, add_global_state)
    elif env_name in _gigastep_registry:
        return make_gigastep_env(env_name, config, add_global_state)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")
