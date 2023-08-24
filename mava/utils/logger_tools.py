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
import warnings
from collections import defaultdict
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Tuple, Union

import neptune.new as neptune
import numpy as np
from colorama import Fore, Style
from omegaconf import DictConfig
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
        self.use_neptune = False

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

    def setup_neptune(self, exp_params: Dict) -> None:
        """Set up neptune logging."""
        self.neptune_logger = NeptuneLogger(
            label="logger",
            tag=exp_params["neptune_tag"],
            name=exp_params["name"],
            exp_params=exp_params,
        )
        self.use_neptune = True

    def log_stat(self, key: str, value: float, t: int) -> None:
        """Log a single stat."""
        self.stats[key].append((t, value))

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_neptune:
            neptune_dict = {key: value}
            self.neptune_logger.write(neptune_dict)

        if self.use_sacred:
            if key in self.sacred_info:
                self.sacred_info[f"{key}_T"].append(t)
                self.sacred_info[key].append(value)
            else:
                self.sacred_info[f"{key}_T"] = [t]
                self.sacred_info[key] = [value]

            self.sacred_run_dict.log_scalar(key, value, t)


class NeptuneLogger(Logger):
    """Logs to the [neptune.ai](https://app.neptune.ai/) platform. The user is expected to have
    their NEPTUNE_API_TOKEN set as an environment variable. This can be done from the Neptune GUI.
    """

    def __init__(
        self,
        label: str,
        tag: str,
        name: Union[str, None] = None,
        exp_params: Union[Dict, None] = None,
        project: str = "InstaDeep/Mava",
        # Logging hardware metrics fails with nvidia migs
        capture_hardware_metrics: bool = True,
    ):
        """Initialise a logger for logging experiment metrics to Neptune.

        Args:
            label: identifier indicating what process the logger was created for.
                eg. executor, evaluator or trainer.
            project: Namespace and project name to be logged to. This will usually be
                something like `f"InstaDeep/{your_project_name}"`.
            tag: a tag for separating experiments from eachother. This is useful
                for grouping, aggregating and plotting experiment results later on.
            name: unique ID from logging to Neptune defaults to the current
                date and time.
            exp_params: all parameters of the current experiment.
            capture_hardware_metrics: whether machine hardware metrics should be
                logged to Neptune.
        """
        self._label = label
        if name:
            self._name = name
        else:
            self._name = str(datetime.now())
        self._exp_params = exp_params
        self._api_token = os.getenv("NEPTUNE_API_TOKEN")
        self._project = project
        self._tag = tag

        self._run = neptune.init(
            name=self._name,
            monitoring_namespace=f"monitoring/{self._label}",
            api_token=self._api_token,
            project=self._project,
            tags=self._tag,
            capture_hardware_metrics=capture_hardware_metrics,
        )
        self._run.params = self._exp_params

    def write(self, values: Any) -> None:  # noqa: CCR001 B028
        """Write values to the logger."""
        try:
            if isinstance(values, dict):
                for key, value in values.items():
                    is_scalar_array = hasattr(value, "shape") and (
                        value.shape == [1] or value.shape == 1 or value.shape == ()
                    )
                    if np.isscalar(value):
                        self.scalar_summary(key, value)
                    elif is_scalar_array:
                        if hasattr(value, "item"):
                            self.scalar_summary(key, value.item())
                        else:
                            self.scalar_summary(key, value)
                    elif hasattr(value, "shape"):
                        self.histogram_summary(key, value)
                    elif isinstance(value, dict):
                        flatten_dict = self._flatten_dict(parent_key=key, dict_info=value)
                        self.write(flatten_dict)
                    elif isinstance(value, tuple) or isinstance(value, list):
                        for index, elements in enumerate(value):
                            self.write({f"{key}_info_{index}": elements})
                    else:
                        warnings.warn(f"Unable to log: {key}, unknown type: {type(value)}")
            elif isinstance(values, tuple) or isinstance(value, list):
                for elements in values:
                    self.write(elements)
            else:
                warnings.warn(
                    f"Unable to log: {values}, unknown type: {type(values)}", stacklevel=2
                )
        except Exception as ex:
            warnings.warn(
                f"Unable to log: {key}, type: {type(value)} , value: {value}" + f" ex: {ex}",
                stacklevel=2,
            )

    def scalar_summary(self, key: str, value: Any) -> None:
        """Log a scalar variable."""
        if self._run:
            self._run[f"{self._label}/{format_key(key)}"].log(value)

    def dict_summary(self, key: str, value: Dict) -> None:
        """Log a dictionary of values."""
        dict_info = self._flatten_dict(parent_key=key, dict_info=value)
        for (k, v) in dict_info.items():
            self.scalar_summary(k, v)

    def histogram_summary(self, key: str, value: np.ndarray) -> None:
        """Log a histogram of the tensor of values."""
        return

    def _flatten_dict(self, parent_key: str, dict_info: Dict, sep: str = "_") -> Dict[str, float]:
        """Flatten a nested dictionary.
        Note:
            Flatten dict, adapted from
            https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
            Converts {'agent_0': {'critic_loss': 0.1, 'policy_loss': 0.2},...}
            to  {'agent_0_critic_loss':0.1,'agent_0_policy_loss':0.1 ,...}
        """
        items: List = []
        for k, v in dict_info.items():
            k = str(k)
            if parent_key:
                new_key = parent_key + sep + k
            else:
                new_key = k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(parent_key=new_key, dict_info=v, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)

    def close(self) -> None:
        """Close the logger."""
        self._run = None

    def stop(self) -> None:
        """Stop the logger."""
        self._run.stop()


def format_key(key: str) -> str:
    """Internal function for formatting keys in Tensorboard format."""
    return key.title().replace("_", "")


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


def get_experiment_path(config: DictConfig, logger_type: str) -> str:
    """Helper function to create the experiment path."""
    exp_path = (
        f"{logger_type}/{config['system_name']}/{config['env_name']}/"
        + f"{config['rware_scenario']['task_name']}/envs_{config['num_envs']}/"
        + f"seed_{config['seed']}"
    )

    return exp_path
