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
from typing import Dict, Optional, Tuple, Union

import jax.numpy as jnp
import numpy as np
from colorama import Fore, Style
from omegaconf import DictConfig

from mava.types import ExperimentOutput
from mava.utils.logger_tools import Logger as LoggerTools


class Logger:
    def __init__(self, config: DictConfig) -> None:
        """Initialise the logger."""
        self.logger = LoggerTools(config)
        self.config = config
        self.anakin_arch = config.arch.arch_name == "anakin"

    def _log_extra_trainer_metrics(self, metrics: Dict, t_env: int = 0) -> str:
        """Log extra trainer metrics."""
        if self.logger.should_log:
            for metric_type in metrics.keys():
                for metric in metrics[metric_type].keys():
                    self.logger.log_stat(
                        f"trainer/{metric_type}/{metric}",
                        metrics[metric_type][metric],
                        t_env,
                    )
        training_time = metrics["speed_info"]["training_time"]
        log_string = (
            f"Timesteps {t_env:07d} | "
            f"Policy Version {metrics['speed_info']['trainer_update_number']} | "
            f"Training Time {float(np.mean(training_time)):.3f}"
        )
        return log_string

    def _log_episode_info(
        self,
        episodes_info: dict,
        t_env: int = 0,
        prefix: str = "evaluator",
        eval_step: Optional[int] = None,
    ) -> Tuple[str, float]:
        """Log the environment's episodes metrics."""
        if self.anakin_arch:
            episodes_return = jnp.ravel(episodes_info["episode_return"])
            episodes_length = jnp.ravel(episodes_info["episode_length"])
        else:
            episodes_return = episodes_info["episode_return"]
            episodes_length = episodes_info["episode_length"]

        if self.logger.should_log:
            self.logger.log_stat(
                f"{prefix}/mean_episode_returns", float(np.mean(episodes_return)), t_env, eval_step
            )
            self.logger.log_stat(
                f"{prefix}/mean_episode_length", float(np.mean(episodes_length)), t_env, eval_step
            )
        log_string = (
            f"Timesteps {t_env:07d} | "
            f"Mean Episode Return {float(np.mean(episodes_return)):.3f} | "
            f"Std Episode Return {float(np.std(episodes_return)):.3f} | "
            f"Max Episode Return {float(np.max(episodes_return)):.3f} | "
            f"Mean Episode Length {float(np.mean(episodes_length)):.3f} | "
            f"Std Episode Length {float(np.std(episodes_length)):.3f} | "
            f"Max Episode Length {float(np.max(episodes_length)):.3f} | "
        )

        if "steps_per_second" in episodes_info.keys():
            steps_per_second = episodes_info["steps_per_second"]
            if self.logger.should_log:
                self.logger.log_stat(
                    f"{prefix}steps_per_second", steps_per_second, t_env, eval_step
                )
            log_string += f"Steps Per Second {steps_per_second:.2e}"

        return log_string, float(np.mean(episodes_return))

    def log_trainer_metrics(
        self,
        experiment_output: Union[ExperimentOutput, Dict],
        t_env: int = 0,
    ) -> None:
        """Log the trainer metrics."""
        # Convert metrics to dict
        if isinstance(experiment_output, ExperimentOutput):
            metrics: Dict = experiment_output._asdict()
            metrics.pop("learner_state")
        else:
            metrics = experiment_output

        if self.anakin_arch:
            loss_info = metrics
            episodes_info = loss_info.pop("episodes_info")
            # Log executor metrics.
            log_string, _ = self._log_episode_info(episodes_info, t_env, "trainer")
        else:
            loss_info = metrics.pop("loss_info")
            # Log extra metrics.
            log_string = self._log_extra_trainer_metrics(metrics, t_env)

        # Log loss metrics.
        if self.logger.should_log:
            for metric in loss_info.keys():
                metric_mean = np.mean(np.array(loss_info[metric]))
                self.logger.log_stat(f"trainer/{metric}", metric_mean, t_env)  # type: ignore

        # Log string.
        log_string += (
            f" | Total Loss {float(np.mean(loss_info['total_loss'])):.3f} | "
            f"Value Loss {float(np.mean(loss_info['value_loss'])):.3f} | "
            f"Loss Actor {float(np.mean(loss_info['loss_actor'])):.3f} | "
            f"Entropy {float(np.mean(loss_info['entropy'])):.3f}"
        )
        self.logger.console_logger.info(
            f"{Fore.MAGENTA}{Style.BRIGHT}TRAINER: {log_string}{Style.RESET_ALL}"
        )

    def log_executor_metrics(
        self,
        metrics: Dict,
        t_env: int = 0,
        device_thread_id: int = 0,
    ) -> None:
        """Log the executor metrics."""
        # Log executor metrics.
        episode_info = metrics.pop("episodes_info")
        log_string, _ = self._log_episode_info(episode_info, t_env, "executor")
        # Log extra metrics.
        if self.logger.should_log:
            for metric_type in metrics.keys():
                for metric in metrics[metric_type].keys():
                    self.logger.log_stat(
                        f"executor/{metric_type}/{metric}",
                        metrics[metric_type][metric],
                        t_env,
                    )
        if device_thread_id == 0:
            rollout_time = metrics["speed_info"]["rollout_time"]
            log_string += f" | Rollout Time {float(np.mean(rollout_time)):.3f}"
            self.logger.console_logger.info(
                f"{Fore.BLUE}{Style.BRIGHT}Executor: {log_string}{Style.RESET_ALL}"
            )

    def log_evaluator_metrics(
        self,
        metrics: Dict,
        t_env: int = 0,
        eval_step: int = 0,
        absolute_metric: bool = False,
    ) -> float:
        """Log the evaluator metrics."""
        if absolute_metric:
            log_string, mean_episode_return = self._log_episode_info(
                metrics, t_env, "absolute", eval_step
            )
            self.logger.console_logger.info(
                f"{Fore.BLUE}{Style.BRIGHT}ABSOLUTE METRIC: {log_string}{Style.RESET_ALL}"
            )
        else:
            log_string, mean_episode_return = self._log_episode_info(
                metrics, t_env, "evaluator", eval_step
            )
            self.logger.console_logger.info(
                f"{Fore.GREEN}{Style.BRIGHT}EVALUATOR: {log_string}{Style.RESET_ALL}"
            )
        return mean_episode_return
