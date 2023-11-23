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
import json
import logging
import os
from typing import Dict, Optional

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
        if cfg["logger"]["use_json"]:
            self._setup_json(cfg)

        self.use_tb = cfg["logger"]["use_tf"]
        self.use_neptune = cfg["logger"]["use_neptune"]
        self.use_json = cfg["logger"]["use_json"]
        self.should_log = bool(
            cfg["logger"]["use_json"] or cfg["logger"]["use_tf"] or cfg["logger"]["use_neptune"]
        )

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

    def _setup_json(self, cfg: Dict) -> None:
        json_exp_path = get_experiment_path(cfg, "json")
        json_logs_path = os.path.join(cfg["base_exp_path"], json_exp_path)
        self.json_logger = JsonWriter(
            path=json_logs_path,
            algorithm_name=cfg["name"],
            task_name=cfg["rware_scenario"]["task_name"],
            environment_name=cfg["env_name"],
            seed=cfg["seed"],
        )

    def log_stat(self, key: str, value: float, t: int, eval_step: Optional[int] = None) -> None:
        """Log a single stat."""

        if self.use_tb:
            self.tb_logger(key, value, t)

        if self.use_neptune:
            self.neptune_logger[key].log(value, step=t)

        if self.use_json and (eval_step is not None):
            self.json_logger.write(t, key, value, eval_step)


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

    run["config"] = stringify_unsupported(cfg)

    return run


def get_experiment_path(config: Dict, logger_type: str) -> str:
    """Helper function to create the experiment path."""
    exp_path = (
        f"{logger_type}/{config['logger']['system_name']}/{config['env']['env_name']}/"
        + f"{config['env']['rware_scenario']['task_name']}"
        + f"/envs_{config['arch']['num_envs']}/seed_{config['system']['seed']}"
    )

    return exp_path


class JsonWriter:
    """
    Writer to create json files for reporting according to marl-eval

    Follows conventions from https://github.com/instadeepai/marl-eval/tree/main#usage-

    Args:
        path (str): where to write the file
        algorithm_name (str): algorithm name
        task_name (str): task name
        environment_name (str): environment name
        seed (int): seed of the experiment

    """

    # TODO(Ruan): Works at the moment and pipes through. But some fixes are still needed. The
    # algorithm name needs to be properly set and the json logger needs a different path so that
    # all seeds from the same exp will log to the same json file. Might be worth keeping this
    # sepearate for now for incase we have distributed experiments.

    def __init__(
        self,
        path: str,
        algorithm_name: str,
        task_name: str,
        environment_name: str,
        seed: int,
    ):
        self.path = path
        self.file_name = "metrics.json"
        self.run_data: Dict = {"absolute_metrics": {}}
        self.data = {
            environment_name: {task_name: {algorithm_name: {f"seed_{seed}": self.run_data}}}
        }
        # Create the direcotry if it doesn't exist
        os.makedirs(self.path, exist_ok=True)

        # Create the file if it doesn't exist
        with open(f"{self.path}/{self.file_name}", "w+") as f:
            json.dump(self.data, f, indent=4)

    def write(self, timestep: int, key: str, value: float, evaluation_step: int) -> None:
        """
        Writes a step into the json reporting file

        Args:
            total_frames (int): total frames collected so far in the experiment
            metrics (dictionary mapping str to tensor): each value is a 1-dim tensor for the metric
                in key of len equal to the number of evaluation episodes for this step.
            evaluation_step (int): the evaluation step

        """
        metrics = {key: [value]}
        step_metrics = {"step_count": timestep}
        # TODO(Ruan): fix the ignore here
        step_metrics.update(metrics)  # type: ignore
        step_str = f"step_{evaluation_step}"
        if step_str in self.run_data:
            self.run_data[step_str].update(step_metrics)
        else:
            self.run_data[step_str] = step_metrics

        # Store the maximum of each metric
        for metric_name in metrics.keys():
            if len(metrics[metric_name]):
                max_metric = max(metrics[metric_name])
                if metric_name in self.run_data["absolute_metrics"]:
                    prev_max_metric = self.run_data["absolute_metrics"][metric_name][0]
                    max_metric = max(max_metric, prev_max_metric)
                self.run_data["absolute_metrics"][metric_name] = [max_metric]

        with open(f"{self.path}/{self.file_name}", "w+") as f:
            json.dump(self.data, f, indent=4)
