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
from typing import Dict, Optional, Protocol, Union

import jax.numpy as jnp
import numpy as np
from colorama import Fore, Style

from mava.types import ExperimentOutput
from mava.utils.logger_tools import Logger


# Not in types.py because we only use it here.
class AnakinLogFn(Protocol):
    def __call__(
        self,
        metrics: ExperimentOutput,
        t_env: int = 0,
        trainer_metric: bool = False,
        absolute_metric: bool = False,
        eval_step: Optional[int] = None,
    ) -> float:
        ...


class SebulbaLogFn(Protocol):
    def __call__(
        self,
        log_type: dict,
        metrics_to_log: dict,
        t_env: int = 0,
    ) -> None:
        ...


def get_logger_tools(  # noqa: CCR001
    logger: Logger, arch_name: str
) -> Union[AnakinLogFn, SebulbaLogFn]:  # noqa: CCR001
    """Get the logger function."""

    def anakin_log(
        metrics: ExperimentOutput,
        t_env: int = 0,
        trainer_metric: bool = False,
        absolute_metric: bool = False,
        eval_step: Optional[int] = None,
    ) -> float:
        """Log the episode returns and lengths.

        Args:
            metrics (Dict): The metrics info.
            t_env (int): The current environment timestep.
            trainer_metric (bool): Whether to log the trainer metric.
            absolute_metric (bool): Whether to log the absolute metric.
            eval_step (int): The count of the current evaluation.
        """
        if absolute_metric:
            prefix = "absolute/"
            episodes_info = metrics.episodes_info
        elif trainer_metric:
            prefix = "trainer/"
            episodes_info = metrics.episodes_info
            total_loss = metrics.total_loss
            value_loss = metrics.value_loss
            loss_actor = metrics.loss_actor
            entropy = metrics.entropy
        else:
            prefix = "evaluator/"
            episodes_info = metrics.episodes_info

        # Flatten metrics info.
        episodes_return = jnp.ravel(episodes_info["episode_return"])
        episodes_length = jnp.ravel(episodes_info["episode_length"])
        steps_per_second = episodes_info["steps_per_second"]

        # Log metrics.
        if logger.should_log:
            logger.log_stat(
                f"{prefix}mean_episode_returns", float(np.mean(episodes_return)), t_env, eval_step
            )
            logger.log_stat(
                f"{prefix}mean_episode_length", float(np.mean(episodes_length)), t_env, eval_step
            )
            logger.log_stat(f"{prefix}steps_per_second", steps_per_second, t_env, eval_step)

            if trainer_metric:
                logger.log_stat(f"{prefix}total_loss", float(np.mean(total_loss)), t_env)
                logger.log_stat(f"{prefix}value_loss", float(np.mean(value_loss)), t_env)
                logger.log_stat(f"{prefix}loss_actor", float(np.mean(loss_actor)), t_env)
                logger.log_stat(f"{prefix}entropy", float(np.mean(entropy)), t_env)

        log_string = (
            f"Timesteps {t_env:07d} | "
            f"Mean Episode Return {float(np.mean(episodes_return)):.3f} | "
            f"Std Episode Return {float(np.std(episodes_return)):.3f} | "
            f"Max Episode Return {float(np.max(episodes_return)):.3f} | "
            f"Mean Episode Length {float(np.mean(episodes_length)):.3f} | "
            f"Std Episode Length {float(np.std(episodes_length)):.3f} | "
            f"Max Episode Length {float(np.max(episodes_length)):.3f} | "
            f"Steps Per Second {steps_per_second:.2e} "
        )

        if absolute_metric:
            logger.console_logger.info(
                f"{Fore.BLUE}{Style.BRIGHT}ABSOLUTE METRIC: {log_string}{Style.RESET_ALL}"
            )
        elif trainer_metric:
            log_string += (
                f"| Total Loss {float(np.mean(total_loss)):.3f} | "
                f"Value Loss {float(np.mean(value_loss)):.3f} | "
                f"Loss Actor {float(np.mean(loss_actor)):.3f} | "
                f"Entropy {float(np.mean(entropy)):.3f}"
            )
            logger.console_logger.info(
                f"{Fore.MAGENTA}{Style.BRIGHT}TRAINER: {log_string}{Style.RESET_ALL}"
            )
        else:
            logger.console_logger.info(
                f"{Fore.GREEN}{Style.BRIGHT}EVALUATOR: {log_string}{Style.RESET_ALL}"
            )

        return float(np.mean(episodes_return))

    def sebulba_log(  # noqa: CCR001
        log_type: dict,
        metrics_to_log: dict,
        t_env: int = 0,
    ) -> None:
        """Log the desired metrics.

        Args:
            log_type (dict): This specifies the types of metrics to be logged, along with
            additional information related to that metric type.
            metrics_to_log (dict): The metrics to log.
            t_env (int): The current environment timestep.
        """
        if "Learner" in log_type.keys():
            total_loss = metrics_to_log["loss_info"]["total_loss"]
            value_loss = metrics_to_log["loss_info"]["value_loss"]
            loss_actor = metrics_to_log["loss_info"]["loss_actor"]
            entropy = metrics_to_log["loss_info"]["entropy"]
            approx_kl = metrics_to_log["loss_info"]["approx_kl"]
            training_time = metrics_to_log["speed_info"]["training_time"]
            log_string = (
                f"Timesteps {t_env:07d} | "
                f"Policy Version {log_type['Learner']['trainer_update_number']} | "
                f"Total Loss {float(np.mean(total_loss)):.3f} | "
                f"Value Loss {float(np.mean(value_loss)):.3f} | "
                f"Loss Actor {float(np.mean(loss_actor)):.3f} | "
                f"Entropy {float(np.mean(entropy)):.3f} | "
                f"Approx KL {float(np.mean(approx_kl)):.3f} | "
                f"Training Time {float(np.mean(training_time)):.3f} | "
            )
            logger.console_logger.info(
                f"{Fore.MAGENTA}{Style.BRIGHT}TRAINER: {log_string}{Style.RESET_ALL}"
            )

            # Log Loss infos
            for metric in metrics_to_log["loss_info"].keys():
                metric_mean = np.mean(metrics_to_log["loss_info"][metric])
                logger.log_stat(f"learner/loss/{metric}", metric_mean, t_env)  # type: ignore

            # Log Speed infos
            logger.log_stat(
                "learner/training_time",
                float(np.mean(metrics_to_log["speed_info"]["training_time"])),
                t_env,
            )

            # Log queue infos
            for metric in metrics_to_log["queue_info"].keys():
                logger.log_stat(
                    f"learner/stats/{metric}", metrics_to_log["queue_info"][metric], t_env
                )

        elif "Executor" in log_type.keys():
            episodes_return = metrics_to_log["episode_info"]["episode_return"]
            episodes_length = metrics_to_log["episode_info"]["episode_length"]
            steps_per_second = metrics_to_log["speed_info"]["sps"]
            rollout_time = metrics_to_log["speed_info"]["rollout_time"]
            if log_type["Executor"]["device_thread_id"] == 0:
                log_string = (
                    f"Timesteps {t_env:07d} | "
                    f"Mean Episode Return {float(np.mean(episodes_return)):.3f} | "
                    f"Max Episode Return {float(np.max(episodes_return)):.3f} | "
                    f"Mean Episode Length {float(np.mean(episodes_length)):.3f} | "
                    f"Std Episode Length {float(np.std(episodes_length)):.3f} | "
                    f"Max Episode Length {float(np.max(episodes_length)):.3f} | "
                    f"Rollout Time {float(np.mean(rollout_time)):.3f} | "
                    f"Steps Per Second {steps_per_second:.2e} "
                )
                logger.console_logger.info(
                    f"{Fore.GREEN}{Style.BRIGHT}Executor: {log_string}{Style.RESET_ALL}"
                )
            # Log Episode infos
            logger.log_stat("executor/mean_episode_returns", float(np.mean(episodes_return)), t_env)
            logger.log_stat("executor/mean_episode_length", float(np.mean(episodes_length)), t_env)

            # Log Speed infos
            logger.log_stat("executor/steps_per_second", steps_per_second, t_env)
            logger.log_stat(
                "executor/rollout_time",
                float(np.mean(metrics_to_log["speed_info"]["rollout_time"])),
                t_env,
            )

            # Log queue infos:
            for metric in metrics_to_log["queue_info"].keys():
                metric_mean = np.mean(metrics_to_log["queue_info"][metric])
                logger.log_stat(f"executor/stats/{metric}", metric_mean, t_env)  # type: ignore
        else:
            # Evaluator
            episodes_return = metrics_to_log["episode_info"]["episode_return"]
            episodes_length = metrics_to_log["episode_info"]["episode_length"]
            steps_per_second = metrics_to_log["speed_info"]["sps"]
            log_string = (
                f"Timesteps {t_env:07d} | "
                f"Mean Episode Return {float(np.mean(episodes_return)):.3f} | "
                f"Max Episode Return {float(np.max(episodes_return)):.3f} | "
                f"Mean Episode Length {float(np.mean(episodes_length)):.3f} | "
                f"Std Episode Length {float(np.std(episodes_length)):.3f} | "
                f"Max Episode Length {float(np.max(episodes_length)):.3f} | "
                f"Steps Per Second {steps_per_second:.2e} "
            )
            logger.console_logger.info(
                f"{Fore.BLUE}{Style.BRIGHT}Evaluator: {log_string}{Style.RESET_ALL}"
            )
            # Log Episode infos
            logger.log_stat(
                "evaluator/mean_episode_returns", float(np.mean(episodes_return)), t_env
            )
            logger.log_stat("evaluator/mean_episode_length", float(np.mean(episodes_length)), t_env)
            # Log Speed infos
            logger.log_stat("evaluator/steps_per_second", steps_per_second, t_env)

    return anakin_log if arch_name == "anakin" else sebulba_log


def logger_setup(config: Dict) -> Union[AnakinLogFn, SebulbaLogFn]:
    """Setup the logger."""
    logger = Logger(config)
    return get_logger_tools(logger, config["arch"]["arch_name"])
