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

import logging
import os
from collections import defaultdict
from typing import Dict, List, Tuple

import neptune
from colorama import Fore, Style
from neptune.integrations.sacred import NeptuneObserver
from sacred import Experiment, observers, utils
from sacred.run import Run


class Logger:
    """Logger class for logging to tensorboard, and sacred.

    Note:
        For the original implementation, please refer to the following link:
        (https://github.com/uoe-agents/epymarl/blob/main/src/utils/logging.py)
    """

    def __init__(self, console_logger: logging.Logger) -> None:
        """Initialise the logger."""
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False

        # defaultdict is used to overcome the problem of missing keys when logging to sacred.
        self.stats: Dict[str, List[Tuple[int, float]]] = defaultdict(lambda: [])

    def setup_tb(self, directory_name: str) -> None:
        """Set up tensorboard logging."""
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value

        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict: Run) -> None:
        """Set up sacred logging."""
        self.sacred_run_dict = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key: str, value: float, t: int) -> None:
        """Log a single stat."""
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred:
            if key in self.sacred_info:
                self.sacred_info[f"{key}_T"].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info[f"{key}_T"] = [t]
                self.sacred_info[key] = [value]

            self.sacred_run_dict.log_scalar(key, value, t)


def should_log(config: Dict) -> bool:
    """Check if the logger should log."""
    return bool(config["use_sacred"] or config["use_tf"] or config["use_neptune"])


def get_python_logger() -> logging.Logger:
    """Set up a custom python logger."""
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        f"{Fore.CYAN}{Style.BRIGHT}%(message)s{Style.RESET_ALL}", "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Set to info to suppress debug outputs.
    logger.setLevel("INFO")

    return logger


def get_neptune_logger(cfg: Dict) -> neptune.Run:
    """Set up neptune logging."""
    name = cfg["name"]
    tags = cfg["neptune_tag"]
    project = cfg["neptune_project"]

    run = neptune.init_run(name=name, project=project, tags=tags)

    del cfg["neptune_tag"]  # neptune doesn't want lists in run params
    run["params"] = cfg

    return run


def get_sacred_exp(cfg: Dict, system_name: str) -> Experiment:
    """Get sacred experiment and set up sacred logging.

    This sets up terminal logging, adds the file observer (to log configs and results as json files)
    and neptune logging (to save logs online) if required.

    Stores files at: base_exp_path/system_name/env_name/task_name/num_envs/seed.
    """
    logger = get_python_logger()
    ex = Experiment("mava", save_git_info=False)
    ex.logger = logger
    ex.captured_out_filter = utils.apply_backspaces_and_linefeeds

    # Set the base path for the experiment.
    cfg["system_name"] = system_name
    exp_path = get_experiment_path(cfg, "sacred")
    file_obs_path = os.path.join(cfg["base_exp_path"], exp_path)

    # add sacred observers
    ex.observers.append(observers.FileStorageObserver.create(file_obs_path))
    if cfg["use_neptune"]:
        run = get_neptune_logger(cfg)
        ex.observers.append(NeptuneObserver(run=run))

    # Add configuration to the experiment.
    ex.add_config(cfg)

    return ex


def get_experiment_path(config: Dict, logger_type: str) -> str:
    """Helper function to create the experiment path."""
    exp_path = (
        f"{logger_type}/{config['system_name']}/{config['env_name']}/"
        + f"{config['rware_scenario']['task_name']}/envs_{config['num_envs']}/"
        + f"seed_{config['seed']}"
    )

    return exp_path
