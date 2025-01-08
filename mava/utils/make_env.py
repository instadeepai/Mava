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

from typing import Tuple, Dict

import gymnasium
import gymnasium as gym
import gymnasium.vector
import gymnasium.wrappers
import jaxmarl
import jumanji
import matrax
from gigastep import ScenarioBuilder
from jaxmarl.environments.smax import map_name_to_scenario
from pufferlib.ocean import Rware
import pufferlib.vector
from psutil import cpu_count
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

from mava.types import MarlEnv
from mava.wrappers import (
    AgentIDWrapper,
    AutoResetWrapper,
    CleanerWrapper,
    ConnectorWrapper,
    GigastepWrapper,
    GymAgentIDWrapper,
    GymRecordEpisodeMetrics,
    GymToJumanji,
    LbfWrapper,
    MabraxWrapper,
    MatraxWrapper,
    MPEWrapper,
    RecordEpisodeMetrics,
    RwareWrapper,
    SmacWrapper,
    SmaxWrapper,
    UoeWrapper,
    PufferAutoResetWrapper,
    PufferToJumanji,
    VectorConnectorWrapper,
    async_multiagent_worker,
)

# Registry mapping environment names to their generator and wrapper classes.
_jumanji_registry = {
    "RobotWarehouse": {"generator": RwareRandomGenerator, "wrapper": RwareWrapper},
    "LevelBasedForaging": {"generator": LbfRandomGenerator, "wrapper": LbfWrapper},
    "Connector": {"generator": ConnectorRandomGenerator, "wrapper": ConnectorWrapper},
    "VectorConnector": {
        "generator": ConnectorRandomGenerator,
        "wrapper": VectorConnectorWrapper,
    },
    "Cleaner": {"generator": CleanerRandomGenerator, "wrapper": CleanerWrapper},
}

# Registry mapping environment names directly to the corresponding wrapper classes.
_matrax_registry = {"Matrax": MatraxWrapper}
_jaxmarl_registry = {"Smax": SmaxWrapper, "MaBrax": MabraxWrapper, "MPE": MPEWrapper}
_gigastep_registry = {"Gigastep": GigastepWrapper}

_gym_registry = {
    "RobotWarehouse": UoeWrapper,
    "LevelBasedForaging": UoeWrapper,
    "SMACLite": SmacWrapper,
}
_puffer_registry = {
    "PufferRobotWarehouse" : Rware
}

def add_extra_wrappers(
    train_env: MarlEnv, eval_env: MarlEnv, config: DictConfig
) -> Tuple[MarlEnv, MarlEnv]:
    # Disable the AgentID wrapper if the environment has implicit agent IDs.
    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)

    if config.system.add_agent_id:
        train_env = AgentIDWrapper(train_env)
        eval_env = AgentIDWrapper(eval_env)

    train_env = AutoResetWrapper(train_env)
    train_env = RecordEpisodeMetrics(train_env)
    eval_env = RecordEpisodeMetrics(eval_env)

    return train_env, eval_env


def make_jumanji_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
    Create a Jumanji environments for training and evaluation.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A tuple of the environments.

    """
    # Config generator and select the wrapper.
    generator = _jumanji_registry[config.env.env_name]["generator"]
    generator = generator(**config.env.scenario.task_config)
    wrapper = _jumanji_registry[config.env.env_name]["wrapper"]

    # Create envs.
    env_config = {**config.env.kwargs, **config.env.scenario.env_kwargs}
    train_env = jumanji.make(config.env.scenario.name, generator=generator, **env_config)
    eval_env = jumanji.make(config.env.scenario.name, generator=generator, **env_config)
    train_env = wrapper(train_env, add_global_state=add_global_state)
    eval_env = wrapper(eval_env, add_global_state=add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)
    return train_env, eval_env


def make_jaxmarl_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
     Create a JAXMARL environment.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A JAXMARL environment.

    """
    kwargs = dict(config.env.kwargs)
    if "smax" in config.env.env_name.lower():
        kwargs["scenario"] = map_name_to_scenario(config.env.scenario.task_name)
    elif "mpe" in config.env.env_name.lower():
        kwargs.update(config.env.scenario.task_config)

    # Create jaxmarl envs.
    train_env: MarlEnv = _jaxmarl_registry[config.env.env_name](
        jaxmarl.make(config.env.scenario.name, **kwargs),
        add_global_state,
    )
    eval_env: MarlEnv = _jaxmarl_registry[config.env.env_name](
        jaxmarl.make(config.env.scenario.name, **kwargs),
        add_global_state,
    )

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)

    return train_env, eval_env


def make_matrax_env(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
    Creates Matrax environments for training and evaluation.

    Args:
    ----
        env_name: The name of the environment to create.
        config: The configuration of the environment.
        add_global_state: Whether to add the global state to the observation.

    Returns:
    -------
        A tuple containing a train and evaluation Matrax environment.

    """
    # Select the Matrax wrapper.
    wrapper = _matrax_registry[config.env.scenario.name]

    # Create envs.
    task_name = config["env"]["scenario"]["task_name"]
    train_env = matrax.make(task_name, **config.env.kwargs)
    eval_env = matrax.make(task_name, **config.env.kwargs)
    train_env = wrapper(train_env, add_global_state)
    eval_env = wrapper(eval_env, add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)
    return train_env, eval_env


def make_gigastep_env(
    config: DictConfig, add_global_state: bool = False
) -> Tuple[MarlEnv, MarlEnv]:
    """
     Create a Gigastep environment.

    Args:
    ----
        env_name (str): The name of the environment to create.
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation. Default False.

    Returns:
    -------
        A tuple of the environments.

    """
    wrapper = _gigastep_registry[config.env.scenario.name]

    kwargs = config.env.kwargs
    scenario = ScenarioBuilder.from_config(config.env.scenario.task_config)

    train_env: MarlEnv = wrapper(scenario.make(**kwargs), has_global_state=add_global_state)
    eval_env: MarlEnv = wrapper(scenario.make(**kwargs), has_global_state=add_global_state)

    train_env, eval_env = add_extra_wrappers(train_env, eval_env, config)
    return train_env, eval_env


def make_gym_env(
    config: DictConfig,
    num_env: int,
    add_global_state: bool = False,
) -> GymToJumanji:
    """
     Create a gymnasium environment.

    Args:
        config (Dict): The configuration of the environment.
        num_env (int) : The number of parallel envs to create.
        add_global_state (bool): Whether to add the global state to the observation. Default False.

    Returns:
        Async environments.
    """
    wrapper = _gym_registry[config.env.env_name]
    config.system.add_agent_id = config.system.add_agent_id & (~config.env.implicit_agent_id)

    def create_gym_env(config: DictConfig, add_global_state: bool = False) -> gymnasium.Env:
        registered_name = f"{config.env.scenario.name}:{config.env.scenario.task_name}"
        env = gym.make(registered_name, disable_env_checker=True, **config.env.kwargs)
        wrapped_env = wrapper(env, config.env.use_shared_rewards, add_global_state)
        return wrapped_env

    envs = gymnasium.vector.AsyncVectorEnv(
        [lambda: create_gym_env(config, add_global_state) for _ in range(num_env)],
        worker=async_multiagent_worker,
    )

    envs = GymToJumanji(envs)
    if config.system.add_agent_id:
        envs = GymAgentIDWrapper(envs)
    envs = GymRecordEpisodeMetrics(envs)

    return envs

def make_puffer_env(
    config: DictConfig,
    num_env: int,
    add_global_state: bool = False,
) -> GymToJumanji:
    """Create and configure a Puffer environment wrapped for Jumanji."""

    env_creator = _puffer_registry[config.env.env_name]

    def create_puffer_env(*env_args, **env_kwargs) -> gymnasium.Env:
        """Wraps the environment with a PufferAutoResetWrapper."""
        return PufferAutoResetWrapper(env_creator, *env_args, **env_kwargs)

    # Determine the number of CPU cores to use.
    #todo: should we move this to config? testing showed that running on multiple cores is slower due to the transfer overhead
    n_cpu_cores = 1  # Using a single CPU core for light environments. 
    if n_cpu_cores >= num_env:
        num_workers, num_parallel_envs = num_env, 1
    else:
        assert num_env % n_cpu_cores == 0, (
            f"The number of environments ({num_env}) must be divisible by the number of CPU cores ({n_cpu_cores})."
        )
        num_workers, num_parallel_envs = n_cpu_cores, num_env // n_cpu_cores

    # Create the vectorized environments.
    env_kwargs = {
        **config['env']['kwargs'],
        **config['env']['scenario']['task_config'],
        "num_envs": num_parallel_envs,
    }
    envs = pufferlib.vector.make(
        create_puffer_env,
        backend=pufferlib.vector.Multiprocessing,
        num_envs=num_workers,
        env_kwargs=env_kwargs,
    )

    # Wrap environments for Jumanji compatibility.
    envs = PufferToJumanji(envs, num_envs=num_env)
    if config.system.add_agent_id:
        envs = GymAgentIDWrapper(envs)
    envs = GymRecordEpisodeMetrics(envs)

    return envs

def sebulba_make(
    config: DictConfig,
    num_env: int,
    add_global_state: bool = False,
    ) -> GymToJumanji:

    env_name = config.env.env_name
    
    if env_name in _puffer_registry:
        return make_puffer_env(config, num_env, add_global_state)
    elif env_name in _gym_registry:
        return make_gym_env(config, num_env, add_global_state)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")



def make(config: DictConfig, add_global_state: bool = False) -> Tuple[MarlEnv, MarlEnv]:
    """
    Create environments for training and evaluation.

    Args:
    ----
        config (Dict): The configuration of the environment.
        add_global_state (bool): Whether to add the global state to the observation.

    Returns:
    -------
        A tuple of the environments.

    """
    env_name = config.env.env_name

    if env_name in _jumanji_registry:
        return make_jumanji_env(config, add_global_state)
    elif env_name in _jaxmarl_registry:
        return make_jaxmarl_env(config, add_global_state)
    elif env_name in _matrax_registry:
        return make_matrax_env(config, add_global_state)
    elif env_name in _gigastep_registry:
        return make_gigastep_env(config, add_global_state)
    else:
        raise ValueError(f"{env_name} is not a supported environment.")
