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

import pytest
from colorama import Fore, Style
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

system_paths = [
    "ppo.ff_ippo",
    "ppo.ff_mappo",
    "ppo.rec_ippo",
    "ppo.rec_mappo",
    "sac.ff_isac",
    "sac.ff_masac",
    "q_learning.rec_iql",
]

discrete_envs = [
    # "gigastep",
    "lbf",
    # "matrax",
    "rware",
    "smax",
]
cnn_envs = ["cleaner", "connector"]
continuous_envs = ["mabrax"]


def _run_system(system_name: str, cfg: DictConfig) -> float:
    """Runs a system."""
    OmegaConf.set_struct(cfg, False)
    cfg.logger.use_neptune = False  # we never want to log these tests to neptune

    system = importlib.import_module(f"mava.systems.{system_name}")
    eval_perf = system.run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}experiment completed{Style.RESET_ALL}")
    return float(eval_perf)


def _get_fast_config(cfg: DictConfig, fast_config: dict) -> DictConfig:
    """Makes the configs use a minimum number of timesteps and evaluations."""
    dconf: dict = OmegaConf.to_container(cfg, resolve=True)
    dconf["system"] |= fast_config["system"]
    dconf["arch"] |= fast_config["arch"]
    cfg = OmegaConf.create(dconf)

    return cfg


@pytest.mark.parametrize("system_path", system_paths)
def test_mava_system(fast_config: dict, system_path: str) -> None:
    """Does a simple test of all mava systems, by checking that everything pipes through."""
    _, system_name = system_path.split(".")

    with initialize(version_base=None, config_path="../mava/configs/"):
        cfg = compose(config_name=f"default_{system_name}")
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("env_name", discrete_envs)
def test_discrete_env(fast_config: dict, env_name: str) -> None:
    """Tests all discrete environments on ff_ippo."""
    system_path = "ppo.ff_ippo"
    system_name = "ff_ippo"

    with initialize(version_base=None, config_path="../mava/configs/"):
        cfg = compose(config_name=f"default_{system_name}", overrides=[f"env={env_name}"])
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("env_name", cnn_envs)
def test_discrete_cnn_env(fast_config: dict, env_name: str) -> None:
    """Tests all cnn environments on ff_ippo."""
    system_path = "ppo.ff_ippo"
    system_name = "ff_ippo"

    overrides = [f"env={env_name}", "network=cnn"]
    with initialize(version_base=None, config_path="../mava/configs/"):
        cfg = compose(config_name=f"default_{system_name}", overrides=overrides)
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("env_name", continuous_envs)
def test_continuous_env(fast_config: dict, env_name: str) -> None:
    """Tests all continuous environments on ff_isac."""
    system_path = "sac.ff_isac"
    system_name = "ff_isac"

    with initialize(version_base=None, config_path="../mava/configs/"):
        cfg = compose(config_name=f"default_{system_name}", overrides=[f"env={env_name}"])
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)
