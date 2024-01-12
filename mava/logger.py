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
from colorama import Fore, Style

from mava.types import ExperimentOutput
from mava.utils.multi_logger import MultiLogger


class Logger:
    def __init__(self, config: Dict) -> None:
        """Initialise the logger."""
        self.logger = MultiLogger(config)
        self.config = config

    def _log_episode_info(
        self,
        episodes_info: dict,
        t_env: int = 0,
        prefix: str = "evaluator",
        eval_step: Optional[int] = None,
    ) -> Tuple[str, float]:
        """Log the environment's episodes metrics."""
        episodes_return = jnp.ravel(episodes_info["episode_return"])
        episodes_length = jnp.ravel(episodes_info["episode_length"])
        steps_per_second = episodes_info["steps_per_second"]

        if self.logger.should_log:
            self.logger.log_stat(
                f"{prefix}/mean_episode_returns", float(jnp.mean(episodes_return)), t_env, eval_step
            )
            self.logger.log_stat(
                f"{prefix}/mean_episode_length", float(jnp.mean(episodes_length)), t_env, eval_step
            )
            self.logger.log_stat(f"{prefix}/steps_per_second", steps_per_second, t_env, eval_step)

        log_string = (
            f"Timesteps {t_env:07d} | "
            f"Mean Episode Return {float(jnp.mean(episodes_return)):.3f} | "
            f"Std Episode Return {float(jnp.std(episodes_return)):.3f} | "
            f"Max Episode Return {float(jnp.max(episodes_return)):.3f} | "
            f"Mean Episode Length {float(jnp.mean(episodes_length)):.3f} | "
            f"Std Episode Length {float(jnp.std(episodes_length)):.3f} | "
            f"Max Episode Length {float(jnp.max(episodes_length)):.3f} | "
            f"Steps Per Second {steps_per_second:.2e}"
        )

        # Add win rate to episodes_info in case it exists.
        if "won_episode" in episodes_info:
            log_string = self._log_win_rate(episodes_info, prefix, log_string, t_env, eval_step)

        return log_string, float(jnp.mean(episodes_return))

    def _log_win_rate(
        self,
        episodes_info: Dict,
        prefix: str,
        log_string: str,
        t_env: int = 0,
        eval_step: Optional[int] = None,
    ) -> str:
        """Log the win rate of the environment's episodes."""
        # Get the number of episodes to evaluate.
        if prefix == "absolute":
            # To measure the absolute metric, we evaluate the best policy
            # found across training over 10 times the evaluation episodes.
            # For more details on the absolute metric please see:
            # https://arxiv.org/abs/2209.10485.
            n_episodes = self.logger.num_eval_episodes * 10
        else:
            n_episodes = self.logger.num_eval_episodes

        # Calculate the win rate.
        n_won_episodes = jnp.sum(episodes_info["won_episode"])
        win_rate = (n_won_episodes / n_episodes) * 100

        if self.logger.should_log:
            self.logger.log_stat(f"{prefix}/win_rate", float(win_rate), t_env, eval_step)

        log_string += f"| Win Rate {win_rate:.2f}%"
        return log_string

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
        loss_info = metrics

        # Log executor metrics.
        episodes_info = loss_info.pop("episodes_info")
        log_string, _ = self._log_episode_info(episodes_info, t_env, "trainer")

        # Log loss metrics.
        if self.logger.should_log:
            for metric in loss_info.keys():
                metric_mean = jnp.mean(jnp.array(loss_info[metric]))
                self.logger.log_stat(f"trainer/{metric}", metric_mean, t_env)  # type: ignore

        # Log string.
        log_string += (
            f" | Total Loss {float(jnp.mean(loss_info['total_loss'])):.3f} | "
            f"Value Loss {float(jnp.mean(loss_info['value_loss'])):.3f} | "
            f"Loss Actor {float(jnp.mean(loss_info['loss_actor'])):.3f} | "
            f"Entropy {float(jnp.mean(loss_info['entropy'])):.3f}"
        )
        self.logger.console_logger.info(
            f"{Fore.MAGENTA}{Style.BRIGHT}TRAINER: {log_string}{Style.RESET_ALL}"
        )

    def log_eval_metrics(
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

    def stop(self) -> None:
        """Stop the logger."""
        if self.logger.use_neptune:
            self.logger.neptune_logger.stop()
