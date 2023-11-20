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
from collections import defaultdict
from typing import Dict, List, Tuple

import neptune
from colorama import Fore, Style
from neptune.utils import stringify_unsupported


class Logger:
    """Logger class for logging to tensorboard, and sacred.

    Note:
        For the original implementation, please refer to the following link:
        (https://github.com/uoe-agents/epymarl/blob/main/src/utils/logging.py)
    """

    def __init__(self, cfg: Dict) -> None:
        """Initialise the logger."""
        self.cfg = cfg
        self.console_logger = get_python_logger()

        self.use_tb = False

        # defaultdict is used to overcome the problem of missing keys when logging to sacred.
        self.stats: Dict[str, List[Tuple[int, float]]] = defaultdict(lambda: [])

    def setup_tb(self, directory_name: str) -> None:
        """Set up tensorboard logging."""
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value

        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_neptune(self) -> None:
        """Set up neptune logging."""
        self.use_neptune = True
        self.neptune_logger = get_neptune_logger(self.cfg)

    def log_stat(self, key: str, value: float, t: int) -> None:
        """Log a single stat."""
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_neptune:
            self.neptune_logger[key].log(value, step=t, wait=True)


def should_log(config: Dict) -> bool:
    """Check if the logger should log."""
    return bool(config["use_tf"] or config["use_neptune"])


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
    run["params"] = stringify_unsupported(cfg)

    return run


def get_experiment_path(config: Dict, logger_type: str) -> str:
    """Helper function to create the experiment path."""
    exp_path = (
        f"{logger_type}/{config['system_name']}/{config['env_name']}/"
        + f"{config['rware_scenario']['task_name']}/envs_{config['num_envs']}/"
        + f"seed_{config['seed']}"
    )

    return exp_path
