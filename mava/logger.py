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
from typing import Dict, Optional, Protocol

import jax.numpy as jnp
import numpy as np
from colorama import Fore, Style

from mava.types import ExperimentOutput
from mava.utils.logger_tools import Logger


# Not in types.py because we only use it here.
class LogFn(Protocol):
    def __call__(
        self,
        metrics: ExperimentOutput,
        t_env: int = 0,
        trainer_metric: bool = False,
        absolute_metric: bool = False,
        eval_step: Optional[int] = None,
    ) -> float:
        ...


def get_logger_tools(logger: Logger) -> LogFn:  # noqa: CCR001
    """Get the logger function."""

    def log(
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

    return log


def logger_setup(config: Dict) -> LogFn:
    """Setup the logger."""
    logger = Logger(config)
    return get_logger_tools(logger)
