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

discrete_envs = ["gigastep", "lbf", "matrax", "rware", "smax"]
cnn_envs = ["cleaner", "connector"]
continuous_envs = ["mabrax"]


def _run_system(system_name: str, cfg: DictConfig) -> float:
    OmegaConf.set_struct(cfg, False)
    cfg.logger.use_neptune = False  # we never want to log these tests to neptune

    system = importlib.import_module(f"mava.systems.{system_name}")
    eval_perf = system.run_experiment(cfg)

    print(f"{Fore.CYAN}{Style.BRIGHT}experiment completed{Style.RESET_ALL}")
    return float(eval_perf)


def _get_fast_config(cfg: DictConfig, fast_config: dict) -> DictConfig:
    # merging fast_config into cfg
    dconf: dict = OmegaConf.to_container(cfg, resolve=True)
    dconf["system"] |= fast_config["system"]
    dconf["arch"] |= fast_config["arch"]
    cfg = OmegaConf.create(dconf)

    return cfg


@pytest.mark.parametrize("system_path", system_paths)
def test_mava_system(fast_config: dict, system_path: str):
    _, system_name = system_path.split(".")

    with initialize(version_base=None, config_path="../mava/configs/"):
        cfg = compose(config_name=f"default_{system_name}")
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("env_name", discrete_envs)
def test_discrete_env(fast_config: dict, env_name: str):
    system_path = "ppo.ff_ippo"
    system_name = "ff_ippo"

    with initialize(version_base=None, config_path="../mava/configs/"):
        cfg = compose(config_name=f"default_{system_name}", overrides=[f"env={env_name}"])
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("env_name", cnn_envs)
def test_discrete_cnn_env(fast_config: dict, env_name: str):
    system_path = "ppo.ff_ippo"
    system_name = "ff_ippo"

    overrides = [f"env={env_name}", "network=cnn"]
    with initialize(version_base=None, config_path="../mava/configs/"):
        cfg = compose(config_name=f"default_{system_name}", overrides=overrides)
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)


@pytest.mark.parametrize("env_name", continuous_envs)
def test_continuous_env(fast_config: dict, env_name: str):
    system_path = "sac.ff_isac"
    system_name = "ff_isac"

    with initialize(version_base=None, config_path="../mava/configs/"):
        cfg = compose(config_name=f"default_{system_name}", overrides=[f"env={env_name}"])
        cfg = _get_fast_config(cfg, fast_config)

    _run_system(system_path, cfg)
