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
"""Logger setup."""

import datetime
import os
from logging import Logger as SacredLogger
from os.path import abspath, dirname
from typing import Callable, Dict

import chex
import jax.numpy as jnp
import numpy as np
from sacred.run import Run

from jax_distribution.utils.logger_tools import Logger


def get_logger_fn(logger: SacredLogger, config: Dict) -> Callable:
    """Get the logger function."""

    def log(
        metrics_info: Dict[str, Dict[str, chex.Array]],
        t_env: int = 0,
        absolute_metric: bool = False,
    ) -> None:
        """Log the episode returns and lengths.

        Args:
            metrics_info (Dict): The metrics info.
            t_env (int): The current timestep.
            absolute_metric (bool): Whether to log the absolute metric.
        """
        if absolute_metric:
            suffix = "_absolute_metric"
        else:
            suffix = ""
        # Flatten metrics info.
        episodes_return = jnp.ravel(metrics_info["episode_return"])
        episodes_length = jnp.ravel(metrics_info["episode_length"])

        # Log metrics.
        if config["USE_SACRED"] or config["USE_TF"]:
            logger.log_stat(
                "mean_test_episode_returns" + suffix,
                float(np.mean(episodes_return)),
                t_env,
            )
            logger.log_stat(
                "mean_test_episode_length" + suffix,
                float(np.mean(episodes_length)),
                t_env,
            )

        log_string = "Timesteps {:07d}".format(t_env) + " "
        log_string += "| Mean Episode Returns {:.3f} ".format(
            float(np.mean(episodes_return))
        )
        log_string += "| Std Episode Returns {:.3f} ".format(
            float(np.std(episodes_return))
        )
        log_string += "| Max Episode Returns {:.3f} ".format(
            float(np.max(episodes_return))
        )
        log_string += "| Mean Episode Length {:.3f} ".format(
            float(np.mean(episodes_length))
        )
        log_string += "| Std Episode Length {:.3f} ".format(
            float(np.std(episodes_length))
        )
        log_string += "| Max Episode Length {:.3f} ".format(
            float(np.max(episodes_length))
        )

        if absolute_metric:
            logger.console_logger.info("ABSOLUTE METRIC:")
        logger.console_logger.info(log_string)

        return float(np.mean(episodes_return))

    return log


def logger_setup(_run: Run, config: Dict, _log: SacredLogger):
    """Setup the logger."""
    logger = Logger(_log)
    unique_token = (
        f"{config['ENV_NAME']}_seed{config['SEED']}_{datetime.datetime.now()}"
    )
    if config["USE_SACRED"]:
        logger.setup_sacred(_run)
    if config["USE_TF"]:
        tb_logs_direc = os.path.join(
            dirname(dirname(abspath(__file__))), "results", "tb_logs"
        )
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
    return get_logger_fn(logger, config)
