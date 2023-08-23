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

"""Logger setup."""
import datetime
import json
import os
import time
from logging import Logger as SacredLogger
from typing import Callable, Dict, Tuple

import jax.numpy as jnp
import numpy as np
from colorama import Fore, Style
from sacred.run import Run

from mava.types import ExperimentOutput
from mava.utils.logger_tools import Logger, get_experiment_path


def get_logger_fn(  # noqa: CCR001
    logger: Logger, _run: Run, config: Dict
) -> Tuple[Callable, Callable]:
    """Get the logger function."""

    def log(  # noqa: CCR001
        metrics: ExperimentOutput,
        t_env: int = 0,
        trainer_metric: bool = False,
        absolute_metric: bool = False,
        seed: int = 0,
    ) -> float:
        """Log the episode returns and lengths.

        Args:
            metrics (Dict): The metrics info.
            t_env (int): The current timestep.
            trainer_metric (bool): Whether to log the trainer metric.
            absolute_metric (bool): Whether to log the absolute metric.
            seed (int): The random seed of the current run.
        """
        if absolute_metric:
            prefix = "Absolute_"
            episodes_info = metrics.episodes_info
        elif trainer_metric:
            prefix = "Trainer_"
            episodes_info = metrics.episodes_info
            total_loss = metrics.total_loss
            value_loss = metrics.value_loss
            loss_actor = metrics.loss_actor
            entropy = metrics.entropy
        else:
            prefix = ""
            episodes_info = metrics.episodes_info

        # Flatten metrics info.
        episodes_return = jnp.ravel(episodes_info["episode_return"])
        episodes_length = jnp.ravel(episodes_info["episode_length"])

        # Log metrics.
        if config["use_sacred"] or config["use_tf"]:
            logger.log_stat(
                prefix.lower() + f"mean_episode_returns_seed_{seed}",
                float(np.mean(episodes_return)),
                t_env,
            )
            logger.log_stat(
                prefix.lower() + f"mean_episode_length_seed_{seed}",
                float(np.mean(episodes_length)),
                t_env,
            )
            if trainer_metric:
                logger.log_stat(f"total_loss_seed_{seed}", float(np.mean(total_loss)), t_env)
                logger.log_stat(f"value_loss_seed_{seed}", float(np.mean(value_loss)), t_env)
                logger.log_stat(f"loss_actor_seed_{seed}", float(np.mean(loss_actor)), t_env)
                logger.log_stat(f"entropy_seed_{seed}", float(np.mean(entropy)), t_env)

        log_string = (
            f"Timesteps {t_env:07d} | "
            f"Mean Episode Return {float(np.mean(episodes_return)):.3f} | "
            f"Std Episode Return {float(np.std(episodes_return)):.3f} | "
            f"Max Episode Return {float(np.max(episodes_return)):.3f} | "
            f"Mean Episode Length {float(np.mean(episodes_length)):.3f} | "
            f"Std Episode Length {float(np.std(episodes_length)):.3f} | "
            f"Max Episode Length {float(np.max(episodes_length)):.3f}"
        )

        if len(config["seeds"]) > 1:
            suffix = f" (SEED {seed})"
        else:
            suffix = ""

        if absolute_metric:
            logger.console_logger.info(
                f"{Fore.BLUE}{Style.BRIGHT}"
                f"ABSOLUTE METRIC{suffix}: {log_string}"
                f"{Style.RESET_ALL}"
            )
        elif trainer_metric:
            log_string += (
                f"| Total Loss {float(np.mean(total_loss)):.3f} | "
                f"Value Loss {float(np.mean(value_loss)):.3f} | "
                f"Loss Actor {float(np.mean(loss_actor)):.3f} | "
                f"Entropy {float(np.mean(entropy)):.3f}"
            )
            logger.console_logger.info(
                f"{Fore.MAGENTA}{Style.BRIGHT}TRAINER{suffix}: {log_string}{Style.RESET_ALL}"
            )
        else:
            logger.console_logger.info(
                f"{Fore.GREEN}{Style.BRIGHT}EVALUATOR{suffix}: {log_string}{Style.RESET_ALL}"
            )

        return float(np.mean(episodes_return))

    def save_json_by_seed() -> None:
        """Split the current json file by seed into separate files."""
        # Make sure the full values are recorded in the JSON file.
        time.sleep(5)
        # Load JSON data from a file
        input_file_path = f"{_run.observers[0].dir}/metrics.json"
        with open(input_file_path, "r") as input_file:
            data = json.load(input_file)

        # Splitting the data and creating separate JSON files for each seed
        seed_data: dict = {}
        for key, value in data.items():
            metric_name_parts = key.split("_seed_")
            if len(metric_name_parts) == 2:
                seed_num = metric_name_parts[1]
                if seed_num not in seed_data:
                    seed_data[seed_num] = {}
                metric_name = metric_name_parts[0]
                seed_data[seed_num][metric_name] = value

        # Create separate JSON files for each seed's metrics
        for seed_num, metrics in seed_data.items():
            output_file_path = f"{_run.observers[0].dir}/metrics_seed_{seed_num}.json"
            with open(output_file_path, "w") as output_file:
                json.dump(metrics, output_file, indent=2)

    return log, save_json_by_seed


def logger_setup(_run: Run, config: Dict, _log: SacredLogger) -> Tuple[Callable, Callable]:
    """Setup the logger."""
    logger = Logger(_log)
    unique_token = f"{datetime.datetime.now()}"
    if config["use_sacred"]:
        logger.setup_sacred(_run)
    if config["use_tf"]:
        exp_path = get_experiment_path(config, "tensorboard")
        tb_logs_path = os.path.join(config["base_exp_path"], f"{exp_path}/{unique_token}")
        logger.setup_tb(tb_logs_path)
    return get_logger_fn(logger, _run, config)
