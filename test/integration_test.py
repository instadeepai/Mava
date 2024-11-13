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

import importlib
import random

import pytest
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

from test.utils import find_replace

# This integration test is not exhaustive, that would be too expensive. This means that not all
# system run all envs, but each env and each system is run at least once.
# For each system we select a random environment to run.
# Then for each environment we select a random system to run.
config_path = "../mava/configs/default"

ppo_systems = [
    "ppo.anakin.ff_ippo",
    "ppo.anakin.ff_mappo",
    "ppo.anakin.rec_ippo",
    "ppo.anakin.rec_mappo",
]

sac_systems = ["sac.anakin.ff_isac", "sac.anakin.ff_masac", "sac.anakin.ff_hasac"]
q_learning_systems = ["q_learning.anakin.rec_iql", "q_learning.anakin.rec_qmix"]
transformer_systems = ["mat.anakin.mat"]
sable_systems = ["sable.anakin.ff_sable", "sable.anakin.rec_sable"]

discrete_envs = ["gigastep", "lbf", "matrax", "rware", "smax", "vector-connector"]
cnn_envs = ["cleaner", "connector"]
continuous_envs = ["mabrax", "mpe"]


def _run_system(system_name: str, cfg: DictConfig) -> float:
    """Runs a system."""
    OmegaConf.set_struct(cfg, False)
    # we never want to log these tests anywhere
    cfg.logger.use_neptune = False
    cfg.logger.use_tb = False
    cfg.logger.use_json = False

    system = importlib.import_module(f"mava.systems.{system_name}")
    eval_perf = system.run_experiment(cfg)

    return float(eval_perf)


def _get_fast_config(cfg: DictConfig, config_modifications: dict) -> DictConfig:
    """Makes the configs use a minimum number of timesteps and evaluations."""
    return OmegaConf.create(
        find_replace(OmegaConf.to_container(cfg, resolve=True), config_modifications)
    )


@pytest.mark.parametrize("system_path", ppo_systems)
def test_ppo_system(fast_config: dict, system_path: str) -> None:
    """Test all ppo systems on random envs."""
    _, _, system_name = system_path.split(".")
    env = random.choice(discrete_envs)

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=f"{system_name}", overrides=[f"env={env}"])
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("system_path", sable_systems)
def test_sable_system(fast_config: dict, system_path: str) -> None:
    """Test all sable systems on random envs."""
    _, _, system_name = system_path.split(".")
    env = random.choice(continuous_envs + discrete_envs)

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=f"{system_name}", overrides=[f"env={env}"])
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("system_path", q_learning_systems)
def test_q_learning_system(fast_config: dict, system_path: str) -> None:
    """Test all Q-Learning systems on random envs."""
    _, _, system_name = system_path.split(".")
    env = random.choice(discrete_envs)

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=f"{system_name}", overrides=[f"env={env}"])
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("system_path", sac_systems)
def test_sac_system(fast_config: dict, system_path: str) -> None:
    """Test all SAC systems on random envs."""
    _, _, system_name = system_path.split(".")
    env = random.choice(continuous_envs)

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=f"{system_name}", overrides=[f"env={env}"])
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("system_path", transformer_systems)
def test_transformer_system(fast_config: dict, system_path: str) -> None:
    """Test transformer systems on random envs."""
    _, _, system_name = system_path.split(".")
    env = random.choice(continuous_envs + discrete_envs)

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=f"{system_name}", overrides=[f"env={env}"])
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("env_name", discrete_envs)
def test_discrete_env(fast_config: dict, env_name: str) -> None:
    """Test all discrete envs on random systems."""
    system_path = random.choice(ppo_systems + q_learning_systems)
    _, _, system_name = system_path.split(".")

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=f"{system_name}", overrides=[f"env={env_name}"])
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("env_name", cnn_envs)
def test_discrete_cnn_env(fast_config: dict, env_name: str) -> None:
    """Test all 2D envs on random systems."""
    system_path = random.choice(ppo_systems)
    _, _, system_name = system_path.split(".")

    network = "cnn" if "ff" in system_name else "rcnn"

    overrides = [f"env={env_name}", f"network={network}"]
    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=f"{system_name}", overrides=overrides)
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("env_name", continuous_envs)
def test_continuous_env(fast_config: dict, env_name: str) -> None:
    """Test all continuous envs on random systems."""
    system_path = random.choice(ppo_systems + sac_systems)
    _, _, system_name = system_path.split(".")
    overrides = [f"env={env_name}"]

    with initialize(version_base=None, config_path=config_path):
        cfg = compose(config_name=f"{system_name}", overrides=overrides)
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)
