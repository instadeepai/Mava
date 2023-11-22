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

import datetime
import logging
import os
from typing import Dict

import neptune
from colorama import Fore, Style
from neptune.utils import stringify_unsupported
from tensorboard_logger import configure, log_value


class Logger:
    """Logger class for logging to tensorboard, and neptune.

    Note:
        For the original implementation, please refer to the following link:
        (https://github.com/uoe-agents/epymarl/blob/main/src/utils/logging.py)
    """

    def __init__(self, cfg: Dict) -> None:
        """Initialise the logger."""
        self.console_logger = get_python_logger()

        if cfg["logger"]["use_tf"]:
            self._setup_tb(cfg)
        if cfg["logger"]["use_neptune"]:
            self._setup_neptune(cfg)

        self.use_tb = cfg["logger"]["use_tf"]
        self.use_neptune = cfg["logger"]["use_neptune"]
        self.should_log = bool(cfg["logger"]["use_tf"] or cfg["logger"]["use_neptune"])

    def _setup_tb(self, cfg: Dict) -> None:
        """Set up tensorboard logging."""
        unique_token = f"{datetime.datetime.now()}"
        exp_path = get_experiment_path(cfg, "tensorboard")
        tb_logs_path = os.path.join(cfg["logger"]["base_exp_path"], f"{exp_path}/{unique_token}")

        configure(tb_logs_path)
        self.tb_logger = log_value

    def _setup_neptune(self, cfg: Dict) -> None:
        """Set up neptune logging."""
        self.neptune_logger = get_neptune_logger(cfg)

    def log_stat(self, key: str, value: float, t: int) -> None:
        """Log a single stat."""

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_neptune:
            self.neptune_logger[key].log(value, step=t)


def get_python_logger() -> logging.Logger:
    """Set up a custom python logger."""
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(f"{Fore.CYAN}{Style.BRIGHT}%(message)s", "%H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Set to info to suppress debug outputs.
    logger.setLevel("INFO")

    return logger


def get_neptune_logger(cfg: Dict) -> neptune.Run:
    """Set up neptune logging."""
    tags = cfg["logger"]["kwargs"]["neptune_tag"]
    project = cfg["logger"]["kwargs"]["neptune_project"]

    run = neptune.init_run(project=project, tags=tags)

    run["params"] = stringify_unsupported(cfg)

    return run


def get_experiment_path(config: Dict, logger_type: str) -> str:
    """Helper function to create the experiment path."""
    exp_path = (
        f"{logger_type}/{config['logger']['system_name']}/{config['env']['env_name']}/"
        + f"{config['env']['rware_scenario']['task_name']}"
        + f"/envs_{config['arch']['num_envs']}/seed_{config['system']['seed']}"
    )

    return exp_path
