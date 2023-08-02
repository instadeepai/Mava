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
from typing import Dict, List

import numpy as np


class Logger:
    """Logger class for logging to tensorboard, sacred, and hdf5."""

    def __init__(self, console_logger: logging.Logger) -> None:
        """Initialise the logger."""
        self.console_logger = console_logger

        self.use_tb = False
        self.use_sacred = False
        self.use_hdf = False
        self.use_custom_metric_logger = False

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
        self._run_obj = sacred_run_dict
        self.sacred_info = sacred_run_dict.info
        self.use_sacred = True

    def log_stat(self, key: str, value: float, t: int, to_sacred: bool = True) -> None:
        """Log a single stat."""
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_sacred and to_sacred:
            if key in self.sacred_info:
                self.sacred_info["{}_T".format(key)].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info["{}_T".format(key)] = [t]
                self.sacred_info[key] = [value]

            self._run_obj.log_scalar(key, value, t)

    def log_list(
        self,
        key: str,
        values: List[float],
        timestamps: List[int] = None,
        to_sacred: bool = True,
    ) -> None:
        """Log a list of stats."""
        if values is not list:
            values = [values]
        if timestamps is None:
            timestamps = range(1, len(values) + 1)
        elif timestamps is not list:
            timestamps = [timestamps]
        elif len(values) != len(timestamps):
            raise ValueError("Number of values and timestamps should match.")

        for value, t in zip(values, timestamps):
            self.stats[key].append((t, value))

            if self.use_tb:
                self.tb_logger(key, value, t)

            if self.use_sacred and to_sacred:
                if key in self.sacred_info:
                    self.sacred_info["{}_T".format(key)].append(t)
                    self.sacred_info[key].append(value)
                else:
                    self.sacred_info["{}_T".format(key)] = [t]
                    self.sacred_info[key] = [value]

                self._run_obj.log_scalar(key, value, t)

    def print_recent_stats(self) -> None:
        """Print the most recent stats."""
        log_str = "Recent Stats | t_env: {:>10} | Episode: {:>8}\n".format(
            *self.stats["episode"][-1]
        )
        i = 0
        for (k, v) in sorted(self.stats.items()):
            if k == "episode":
                continue
            i += 1
            window = 5 if k != "epsilon" else 1
            try:
                item = "{:.4f}".format(np.mean([x[1] for x in self.stats[k][-window:]]))
            except:
                item = "{:.4f}".format(
                    np.mean([x[1].item() for x in self.stats[k][-window:]])
                )
            log_str += "{:<25}{:>8}".format(k + ":", item)
            log_str += "\n" if i % 4 == 0 else "\t"
        self.console_logger.info(log_str)

    def log_custom_metrics(self, metric_dict: Dict, t: int) -> None:
        """Made specifically for storing our custom metrics in a faster way."""

        # Note: Custom metrics must be a dictionary of the form {metric_name: [metric_values]}
        # where [metric_values] is a list of metric values at each logging step.
        self.custom_metrics_logger.write(metric_dict, t)


def get_logger() -> logging.Logger:
    """Set up a custom logger."""
    logger = logging.getLogger()
    logger.handlers = []
    ch = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(levelname)s %(asctime)s] %(name)s %(message)s", "%H:%M:%S"
    )
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # Set to info to suppress debug outputs.
    logger.setLevel("INFO")

    return logger


def recursive_dict_update(d: Dict, u: Dict) -> Dict:
    """Recursively update a dictionary."""
    for k, v in u.items():
        if isinstance(v, collections.Mapping):
            d[k] = recursive_dict_update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def config_copy(config: Dict) -> Dict:
    """Deep copy a config."""
    if isinstance(config, dict):
        return {k: config_copy(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [config_copy(v) for v in config]
    else:
        return deepcopy(config)
