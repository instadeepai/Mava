# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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
import collections
import logging
from collections import defaultdict
from copy import deepcopy
from typing import Dict

from colorama import Fore, Style


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

        self.tb_logger = None
        self.sacred_run_dict = None
        self.sacred_info = None

        # defaultdict is used to overcome the problem of missing keys when logging to sacred.
        self.stats = defaultdict(lambda: [])

    def setup_tb(self, directory_name: str) -> None:
        """Set up tensorboard logging."""
        # Import here so it doesn't have to be installed if you don't use it
        from tensorboard_logger import configure, log_value

        configure(directory_name)
        self.tb_logger = log_value
        self.use_tb = True

    def setup_sacred(self, sacred_run_dict: Dict) -> None:
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


def get_logger() -> logging.Logger:
    """Set up a custom logger."""
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


def config_copy(config: Dict) -> Dict:
    """Deep copy a config."""
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)
