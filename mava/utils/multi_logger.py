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

import abc
import json
import logging
import os
import time
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import jax
import neptune
from colorama import Fore, Style
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig
from pandas.io.json._normalize import _simple_json_normalize as flatten_dict
from tensorboard_logger import configure, log_value


class LogEvent(Enum):
    ACT = "actor"
    TRAIN = "trainer"
    EVAL = "evaluator"
    ABSOLUTE = "absolute metric"
    MISC = "misc"


class MavaLogger(abc.ABC):
    @abc.abstractmethod
    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        """Initialise the logger."""
        ...

    @abc.abstractmethod
    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        raise NotImplementedError

    def log_dict(self, data: Dict, step: int, eval_step: int, event: LogEvent) -> None:
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep="/")

        for key, value in data.items():
            self.log_stat(key, value, step, eval_step, event)

    def stop(self) -> None:
        """Stop the logger."""
        ...


class NeptuneLogger(MavaLogger):
    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        """Set up neptune logging."""
        tags = list(cfg.logger.kwargs.neptune_tag)
        project = cfg.logger.kwargs.neptune_project

        self.logger = neptune.init_run(project=project, tags=tags)

        self.logger["config"] = stringify_unsupported(cfg)

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent):
        t = step if event != LogEvent.EVAL else eval_step
        self.logger[f"{event.value}/{key}"].log(value, step=t)

    def stop(self) -> None:
        return self.logger.stop()


class TensorboardLogger(MavaLogger):
    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        """Set up tensorboard logging."""
        tb_exp_path = get_logger_path(cfg, "tensorboard")
        tb_logs_path = os.path.join(cfg.logger.base_exp_path, f"{tb_exp_path}/{unique_token}")

        configure(tb_logs_path)
        self.log = log_value

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent):
        t = step if event != LogEvent.EVAL else eval_step
        self.log(f"{event.value}/{key}", value, t)


class JsonLogger(MavaLogger):
    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        """Set up json logging."""
        json_exp_path = get_logger_path(cfg, "json")
        json_logs_path = os.path.join(cfg.logger.base_exp_path, f"{json_exp_path}/{unique_token}")

        # if a custom path is specified, use that instead
        if cfg.logger.kwargs.json_path is not None:
            json_logs_path = os.path.join(
                cfg.logger.base_exp_path, "json", cfg.logger.kwargs.json_path
            )

        self.logger = JsonWriter(
            path=json_logs_path,
            algorithm_name=cfg.logger.system_name,
            task_name=cfg.env.scenario.task_name,
            environment_name=cfg.env.env_name,
            seed=cfg.system.seed,
        )

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent):
        # JsonWriter can't serialize jax arrays
        value = value.item() if isinstance(value, jax.Array) else value
        self.logger.write(step, f"{event.value}/{key}", value, eval_step)


class ConsoleLogger(MavaLogger):
    _EVENT_COLOURS = {
        LogEvent.TRAIN: Fore.MAGENTA,
        LogEvent.EVAL: Fore.GREEN,
        LogEvent.ABSOLUTE: Fore.BLUE,
        LogEvent.ACT: Fore.CYAN,
        LogEvent.MISC: Fore.YELLOW,
    }

    def __init__(self, cfg: DictConfig, unique_token: str) -> None:
        """Set up console logging."""
        # todo: this writes to outputs/... we should put it at the same path in get_logger_path
        self.logger = logging.getLogger()
        # todo: necessary?
        self.logger.handlers = []
        # todo: necessary?
        ch = logging.StreamHandler()
        formatter = logging.Formatter(f"{Fore.CYAN}{Style.BRIGHT}%(message)s", "%H:%M:%S")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)
        # Set to info to suppress debug outputs.
        self.logger.setLevel("INFO")

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent):
        colour = self._EVENT_COLOURS[event]
        key = key.replace("_", " ").capitalize()
        self.logger.info(
            f"{colour}{Style.BRIGHT}{event.value.upper()} - {key}: {value}{Style.RESET_ALL}"
        )

    def log_dict(self, data: Dict, step: int, eval_step: int, event: LogEvent):
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep=" ")

        colour = self._EVENT_COLOURS[event]
        # Replace underscores with spaces and capitalise keys. Round values to 3 decimal places.
        log_str = " | ".join(
            [f"{k.replace('_', ' ').capitalize()}: {v:.3f}" for k, v in data.items()]
        )

        self.logger.info(
            f"{colour}{Style.BRIGHT}{event.value.upper()} - {log_str}{Style.RESET_ALL}"
        )


# todo: Docstrings
class MultiLogger(MavaLogger):
    def __init__(self, loggers: List[MavaLogger]) -> None:
        self.loggers = loggers

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        for logger in self.loggers:
            logger.log_stat(key, value, step, eval_step, event)

    def log_dict(self, data: Dict, step: int, eval_step: int, event: LogEvent):
        for logger in self.loggers:
            logger.log_dict(data, step, eval_step, event)


def make_logger(cfg: DictConfig) -> MavaLogger:
    loggers = []
    unique_token = datetime.now().strftime("%Y%m%d%H%M%S")

    if cfg.logger.use_neptune:
        loggers.append(NeptuneLogger(cfg, unique_token))
    if cfg.logger.use_tb:
        loggers.append(TensorboardLogger(cfg, unique_token))
    if cfg.logger.use_json:
        loggers.append(JsonLogger(cfg, unique_token))
    if cfg.logger.use_console:
        loggers.append(ConsoleLogger(cfg, unique_token))

    return MultiLogger(loggers)


def get_logger_path(config: DictConfig, logger_type: str) -> str:
    """Helper function to create the experiment path."""
    return (
        f"{logger_type}/{config.logger.system_name}/{config.env.env_name}/"
        + f"{config.env.scenario.task_name}"
        + f"/envs_{config.arch.num_envs}/seed_{config.system.seed}"
    )


# todo: move this to marl-eval
class JsonWriter:
    """
    Writer to create json files for reporting experiment results according to marl-eval

    Follows conventions from https://github.com/instadeepai/marl-eval/tree/main#usage-
    This writer was adapted from the implementation found in BenchMARL. For the original
    implementation please see https://tinyurl.com/2t6fy548

    Args:
        path (str): where to write the file
        algorithm_name (str): algorithm name
        task_name (str): task name
        environment_name (str): environment name
        seed (int): random seed of the experiment
    """

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

        # If the file already exists, load it
        if os.path.isfile(f"{self.path}/{self.file_name}"):
            with open(f"{self.path}/{self.file_name}", "r") as f:
                data = json.load(f)

        else:
            # Create the logging directory if it doesn't exist
            os.makedirs(self.path, exist_ok=True)
            data = {}

        # Merge the existing data with the new data
        self.data = data
        if environment_name not in self.data:
            self.data[environment_name] = {}
        if task_name not in self.data[environment_name]:
            self.data[environment_name][task_name] = {}
        if algorithm_name not in self.data[environment_name][task_name]:
            self.data[environment_name][task_name][algorithm_name] = {}
        self.data[environment_name][task_name][algorithm_name][f"seed_{seed}"] = self.run_data

        with open(f"{self.path}/{self.file_name}", "w") as f:
            json.dump(self.data, f, indent=4)

    def write(
        self,
        timestep: int,
        key: str,
        value: float,
        evaluation_step: Optional[int] = None,
    ) -> None:
        """
        Writes a step to the json reporting file

        Args:
            timestep (int): the current environment timestep
            key (str): the metric that should be logged
            value (str): the value of the metric that should be logged
            evaluation_step (int): the evaluation step
        """

        current_time = time.time()

        # This will ensure the first logged time is 0, which avoids taking compilation into account
        # when plotting downstream.
        if evaluation_step == 0:
            self.start_time = current_time

        logging_prefix, *metric_key = key.split("/")
        metric_key = "/".join(metric_key)

        metrics = {metric_key: [value]}

        if logging_prefix == "evaluator":
            step_metrics = {"step_count": timestep, "elapsed_time": current_time - self.start_time}
            step_metrics.update(metrics)  # type: ignore
            step_str = f"step_{evaluation_step}"
            if step_str in self.run_data:
                self.run_data[step_str].update(step_metrics)
            else:
                self.run_data[step_str] = step_metrics

        # Store the absolute metrics
        if logging_prefix == "absolute":
            self.run_data["absolute_metrics"].update(metrics)

        with open(f"{self.path}/{self.file_name}", "w") as f:
            json.dump(self.data, f, indent=4)
