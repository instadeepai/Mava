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
import logging
import os
import zipfile
from datetime import datetime
from enum import Enum
from os import PathLike
from typing import Callable, ClassVar, Dict, List, Union

import hydra
import jax
import neptune
import numpy as np
from colorama import Fore, Style
from etils.epath import Path
from jax import tree
from jax.typing import ArrayLike
from marl_eval.json_tools import JsonLogger as MarlEvalJsonLogger
from neptune.utils import stringify_unsupported
from omegaconf import DictConfig, OmegaConf
from pandas.io.json._normalize import _simple_json_normalize as flatten_dict
from rich.pretty import pprint
from tensorboard_logger import configure, log_value

from mava.types import Metrics


class LogEvent(Enum):
    ACT = "actor"
    TRAIN = "trainer"
    EVAL = "evaluator"
    ABSOLUTE = "absolute"
    MISC = "misc"


def winrate_custom_metric(metrics: Metrics) -> Metrics:
    """Calculate win rate from episode metrics.

    This is an example of a possible custom metric function. Define a new function for your own
    custom metrics. A custom metrics function needs to take in metrics and process the metrics into
    a form that will then be logged.

    In this example we process the 'won_episode' and 'is_terminal_step' metrics into a winrate by
    dividing the number of wins by the number of episodes.

    Args:
    ----
        metrics: Dictionary of metrics containing 'won_episode' and 'is_terminal_step'.
        config: The system config.

    Returns:
    -------
        Dictionary with added 'win_rate' metric and 'won_episode' removed.
    """
    if "won_episode" not in metrics:
        return metrics

    # Count the number of terminal steps to determine episode count
    is_terminal_steps = metrics.get("is_terminal_step", np.array([]))
    n_episodes: int = np.sum(is_terminal_steps)

    # If no episodes were completed, return unchanged metrics
    if n_episodes == 0:
        return metrics

    # Calculate the win rate
    n_won_episodes: int = np.sum(metrics["won_episode"])
    win_rate: float = (n_won_episodes / n_episodes) * 100

    # Update metrics
    metrics["win_rate"] = win_rate
    metrics.pop("won_episode")

    return metrics


class MavaLogger:
    def __init__(
        self,
        config: DictConfig,
        custom_metrics_fn: Callable[[Metrics], Metrics] = winrate_custom_metric,
    ) -> None:
        """The main logger for Mava systems.

        Thin wrapper around the MultiLogger that is able to describe arrays of metrics
        and calculate environment specific metrics if required (e.g winrate).

        Args:
        ____
            config: The system config.
            custom_metrics_fn: A function that can process the metrics to produce custom metrics.
                This function takes in all metrics and can use data over a rollout to procudce
                environment specific metrics, which it then adds to the metrics dictionary.
                For example a win-rate.
        """
        self.logger: BaseLogger = _make_multi_logger(config)
        self.cfg = config
        self.custom_metrics_fn = custom_metrics_fn

    def log_config(self, config: Dict | None = None) -> None:
        """Log configuration dictionary.

        Args:
        ----
            config: Configuration to log. If None, uses the config provided during initialization.
        """
        cfg = config if config is not None else OmegaConf.to_container(self.cfg, resolve=True)
        self.logger.log_config(cfg)  # type: ignore

    def log(self, metrics: Metrics, t: int, t_eval: int, event: LogEvent) -> None:
        """Log a dictionary of metrics at a given timestep.

        Args:
        ----
            metrics (Metrics): dictionary of metrics to log.
            t (int): the current timestep.
            t_eval (int): the number of previous evaluations.
            event (LogEvent): the event that the metrics are associated with.

        """
        # Apply custom metrics calculation
        metrics = self.custom_metrics_fn(metrics)

        # Remove the is_terminal_step flag if it exists since we're done with it
        if "is_terminal_step" in metrics:
            metrics.pop("is_terminal_step")

        if event == LogEvent.TRAIN:
            # We only want to log mean losses, max/min/std don't matter.
            metrics = tree.map(np.mean, metrics)
        else:
            # {metric1_name: [metrics], metric2_name: ...} ->
            # {metric1_name: {mean: metric, max: metric, ...}, metric2_name: ...}
            metrics = tree.map(describe, metrics)

        self.logger.log_dict(metrics, t, t_eval, event)

    def stop(self) -> None:
        """Stop the logger."""
        self.logger.stop()


class BaseLogger(abc.ABC):
    @abc.abstractmethod
    def __init__(self, base_exp_path: PathLike, unique_token: str, system_name: str) -> None:
        pass

    @abc.abstractmethod
    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        """Log a single metric."""
        raise NotImplementedError

    @abc.abstractmethod
    def log_config(self, config: Dict) -> None:
        """Log configuration dictionary.

        Args:
        ----
            config: Configuration dictionary to log.
        """
        raise NotImplementedError

    def log_dict(self, data: Metrics, step: int, eval_step: int, event: LogEvent) -> None:
        """Log a dictionary of metrics."""
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep="/")

        for key, value in data.items():
            self.log_stat(key, value, step, eval_step, event)

    def stop(self) -> None:
        """Stop the logger."""
        return None


class MultiLogger(BaseLogger):
    def __init__(self, loggers: List[BaseLogger]) -> None:
        """Logger that can log to multiple loggers at once."""
        self.loggers = loggers

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        for logger in self.loggers:
            logger.log_stat(key, value, step, eval_step, event)

    def log_config(self, config: Dict) -> None:
        for logger in self.loggers:
            logger.log_config(config)

    def log_dict(self, data: Metrics, step: int, eval_step: int, event: LogEvent) -> None:
        for logger in self.loggers:
            logger.log_dict(data, step, eval_step, event)

    def stop(self) -> None:
        for logger in self.loggers:
            logger.stop()


class NeptuneLogger(BaseLogger):
    def __init__(
        self,
        base_exp_path: PathLike,
        unique_token: str,
        system_name: str,
        project: str,
        tag: list[str],
        group_tag: list[str],
        detailed_logging: bool,
        architecture_name: str,
        upload_json_data: bool,
        run_id: str | None = None,
    ) -> None:
        """
        Initialize neptune.ai logger for experiment tracking.

        Args:
            base_exp_path: Base path where all logs are stored.
            unique_token: Unique identifier string for this run.
            system_name: Name of the system/algorithm being logged.
            project: neptune.ai project name.
            tag: List of tags for the neptune.ai experiment.
            group_tag: List of group tags - useful for keeping track of a group of experiments.
            detailed_logging: Whether to log detailed metrics (incl. std/min/max).
            architecture_name: Name of the architecture [anakin | sebulba].
            upload_json_data: Whether to upload JSON data to neptune.ai.
            run_id: ID of the run you wish to resume - None if you don't want to resume the run.
                Note this will overwrite the run if you restart the step from 0.
        """
        # async logging leads to deadlocks in sebulba
        mode = "async" if architecture_name == "anakin" else "sync"

        if run_id is not None:
            self.logger = neptune.init_run(with_id=run_id, project=project, mode=mode)
        else:
            self.logger = neptune.init_run(project=project, tags=list(tag), mode=mode)
            self.logger["sys/group_tags"].add(list(group_tag))

        self.detailed_logging = detailed_logging
        self.upload_json_data = upload_json_data

        # Store json path for uploading json data to Neptune.
        json_exp_path = get_logger_path(system_name, "json")
        self.json_file_path = Path(base_exp_path, json_exp_path, unique_token, "metrics.json")
        self.unique_token = unique_token

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        # Main metric if it's the mean of a list of metrics (ends with '/mean')
        # or it's a single metric doesn't contain a '/'.
        is_main_metric = "/" not in key or key.endswith("/mean")
        # If we're not detailed logging (logging everything) then make sure it's a main metric.
        if not self.detailed_logging and not is_main_metric:
            return

        value = value.item() if isinstance(value, (jax.Array, np.ndarray)) else value
        self.logger[f"{event.value}/{key}"].log(value, step=step)

    def log_config(self, config: Dict) -> None:
        self.logger["config"] = stringify_unsupported(config)

    def stop(self) -> None:
        if self.upload_json_data:
            self._zip_and_upload_json()
        self.logger.stop()

    def _zip_and_upload_json(self) -> None:
        # Create the zip file path by replacing '.json' with '.zip'
        zip_file_path = self.json_file_path.with_suffix(".zip").as_posix()

        # Create a zip file containing the specified JSON file
        with zipfile.ZipFile(zip_file_path, "w", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(self.json_file_path, arcname=self.json_file_path.name)

        self.logger[f"metrics/metrics_{self.unique_token}"].upload(zip_file_path)


class TensorboardLogger(BaseLogger):
    def __init__(self, base_exp_path: PathLike, unique_token: str, system_name: str) -> None:
        """
        Initialize TensorBoard logger for visualization.

        Args:
            base_exp_path: Base path where logs will be stored
            unique_token: Unique identifier string for this run
            system_name: Name of the system/algorithm being logged
        """
        tb_exp_path = get_logger_path(system_name, "tensorboard")
        tb_logs_path = os.path.join(base_exp_path, Path(tb_exp_path, unique_token))

        configure(tb_logs_path)
        self.log = log_value

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        t = step if event != LogEvent.EVAL else eval_step
        self.log(f"{event.value}/{key}", value, t)

    def log_config(self, config: Dict) -> None: ...


class JsonLogger(BaseLogger):
    # These are the only metrics that marl-eval needs to plot.
    _METRICS_TO_LOG: ClassVar[List[str]] = ["episode_return/mean", "win_rate", "steps_per_second"]

    def __init__(
        self,
        base_exp_path: PathLike,
        unique_token: str,
        system_name: str,
        path: PathLike | None,
        task_name: str,
        env_name: str,
        seed: int,
    ) -> None:
        """
        Initialize JSON logger for marl-eval compatibility.

        Args:
            base_exp_path: Base path where all logs are stored.
            unique_token: Unique identifier string for this run.
            system_name: Name of the system/algorithm being logged.
            path: Optional custom path for JSON logs (if None, uses default).
            task_name: Name of the scenario/task being evaluated.
            env_name: Name of the environment.
            seed: Random seed used in the experiment.
        """
        json_exp_path = get_logger_path(system_name, "json")
        json_logs_path = Path(base_exp_path, json_exp_path, unique_token)
        # if a custom path is specified, use that instead
        if path is not None:
            json_logs_path = Path(base_exp_path, "json", path)

        self.logger = MarlEvalJsonLogger(
            path=json_logs_path,
            algorithm_name=system_name,
            task_name=task_name,
            environment_name=env_name,
            seed=seed,
        )

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        # Only write key if it's in the list of metrics to log.

        if key not in self._METRICS_TO_LOG:
            return

        # The key is in the format <metric_name>/<aggregation_fn> so we need to change it to:
        # <agg fn>_<metric_name>
        if "/" in key:
            key = "_".join(reversed(key.split("/")))

        # JsonWriter can't serialize jax arrays
        value = value.item() if isinstance(value, jax.Array) else value

        # We only want to log evaluation metrics to the json logger
        if event == LogEvent.ABSOLUTE or event == LogEvent.EVAL:
            self.logger.write(step, key, value, eval_step, event == LogEvent.ABSOLUTE)

    def log_config(self, config: Dict) -> None: ...


class ConsoleLogger(BaseLogger):
    _EVENT_COLOURS: ClassVar[Dict[LogEvent, str]] = {
        LogEvent.TRAIN: Fore.MAGENTA,
        LogEvent.EVAL: Fore.GREEN,
        LogEvent.ABSOLUTE: Fore.BLUE,
        LogEvent.ACT: Fore.CYAN,
        LogEvent.MISC: Fore.YELLOW,
    }

    def __init__(self, base_exp_path: PathLike, unique_token: str, system_name: str) -> None:
        """
        Initialize console logger for stdout output.

        Args:
            base_exp_path: Base path for all experiment logs (not used directly).
            unique_token: Unique identifier string for this run.
            system_name: Name of the system/algorithm being logged.
        """
        self.logger = logging.getLogger()

        self.logger.handlers = []

        ch = logging.StreamHandler()
        formatter = logging.Formatter(f"{Fore.CYAN}{Style.BRIGHT}%(message)s", "%H:%M:%S")
        ch.setFormatter(formatter)
        self.logger.addHandler(ch)

        # Set to info to suppress debug outputs.
        self.logger.setLevel("INFO")

    def log_stat(self, key: str, value: float, step: int, eval_step: int, event: LogEvent) -> None:
        colour = self._EVENT_COLOURS[event]

        # Replace underscores with spaces and capitalise keys.
        key = key.replace("_", " ").capitalize()
        self.logger.info(
            f"{colour}{Style.BRIGHT}{event.value.upper()} - {key}: {value:.3f}{Style.RESET_ALL}"
        )

    def log_dict(self, data: Metrics, step: int, eval_step: int, event: LogEvent) -> None:
        # in case the dict is nested, flatten it.
        data = flatten_dict(data, sep=" ")

        colour = self._EVENT_COLOURS[event]
        # Replace underscores with spaces and capitalise keys.
        keys = [k.replace("_", " ").capitalize() for k in data.keys()]
        # Round values to 3 decimal places if they are floats.
        values = []
        for value in data.values():
            value = value.item() if isinstance(value, jax.Array) else value
            values.append(f"{value:.3f}" if isinstance(value, float) else str(value))
        log_str = " | ".join([f"{k}: {v}" for k, v in zip(keys, values, strict=True)])

        self.logger.info(
            f"{colour}{Style.BRIGHT}{event.value.upper()} - {log_str}{Style.RESET_ALL}"
        )

    def log_config(self, config: Metrics) -> None:
        colour = self._EVENT_COLOURS[LogEvent.MISC]
        self.logger.info(f"{colour}{Style.BRIGHT}CONFIG{Style.RESET_ALL}")
        pprint(config)


def _make_multi_logger(cfg: DictConfig) -> MultiLogger:
    """Instantiate only enabled loggers and remove the 'enabled' flag."""
    unique_token = datetime.now().strftime("%Y%m%d%H%M%S")

    if (
        cfg.logger.loggers.neptune.enabled
        and cfg.logger.loggers.json.enabled
        and cfg.logger.loggers.neptune.upload_json_data
        and cfg.logger.loggers.json.path
    ):
        raise ValueError(
            "Cannot upload json data to Neptune when `json_path` is set in the base logger config. "
            "This is because each subsequent run will create a larger json file which will use "
            "unnecessary storage. Either set `upload_json_data: false` if you don't want to "
            "upload your json data but store a large file locally or set `json_path: ~` in "
            "the base logger config."
        )
    loggers: List[BaseLogger] = []
    for _logger_config in cfg.logger.loggers.values():
        logger_config = dict(_logger_config)  # Create a copy to avoid modifying the original

        # Check if logger is enabled (default to True if not specified)
        if logger_config.pop("enabled", True):
            logger = hydra.utils.instantiate(
                logger_config,
                base_exp_path=cfg.logger.base_exp_path,
                unique_token=unique_token,
                system_name=cfg.logger.system_name,
            )
            loggers.append(logger)

    return MultiLogger(loggers)


def get_logger_path(system_name: str, logger_type: str) -> Path:
    """Helper function to create the experiment path."""
    return Path(logger_type, system_name)


def describe(x: ArrayLike) -> Union[Dict[str, ArrayLike], ArrayLike]:
    """Generate summary statistics for an array of metrics (mean, std, min, max)."""
    if not isinstance(x, (jax.Array, np.ndarray)) or x.ndim == 0:
        return x

    # np instead of jnp because we don't jit here
    return {"mean": np.mean(x), "std": np.std(x), "min": np.min(x), "max": np.max(x)}
