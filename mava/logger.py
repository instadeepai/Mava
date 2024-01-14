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
from typing import Dict

import chex
import jax
import jax.numpy as jnp
import numpy as np
from jax.typing import ArrayLike
from omegaconf import DictConfig

from mava.utils.multi_logger import LogEvent, MavaLogger, make_logger


def describe(x: ArrayLike):
    if not isinstance(x, jax.Array) or x.size <= 1:
        return x

    # np instead of jnp because we don't jit here
    return {"mean": np.mean(x), "std": np.std(x), "min": np.min(x), "max": np.max(x)}


# todo: merge with multilogger?
class Logger:
    """Thin wrapper around the MultiLogger to describe lists of metrics
    and calculate extra metrics if required (e.g winrate).
    """

    def __init__(self, config: DictConfig) -> None:
        self.logger: MavaLogger = make_logger(config)

    def log(self, metrics: Dict, t: int, t_eval: int, event: LogEvent) -> None:
        # Ideally we want to avoid special metrics like this as much as possible.
        # Might be better to calculate this outside as we want to keep the number of these
        # if statements to a minimum.
        if "won_episode" in metrics:
            metrics = self.calc_winrate(metrics, event)

        # {metrics_name: metric} -> {metrics_name: {mean: metric, ...}}
        metrics = jax.tree_map(describe, metrics)
        self.logger.log_dict(metrics, t, t_eval, event)

    def calc_winrate(self, episode_metrics: Dict, event: LogEvent) -> Dict:
        """Log the win rate of the environment's episodes."""
        # Get the number of episodes to evaluate.
        if event == LogEvent.ABSOLUTE:
            # To measure the absolute metric, we evaluate the best policy
            # found across training over 10 times the evaluation episodes.
            # For more details on the absolute metric please see:
            # https://arxiv.org/abs/2209.10485.
            n_episodes = self.logger.num_eval_episodes * 10
        else:
            n_episodes = self.logger.num_eval_episodes

        # Calculate the win rate.
        n_won_episodes = jnp.sum(episode_metrics["won_episode"])
        win_rate = (n_won_episodes / n_episodes) * 100

        episode_metrics["win_rate"] = win_rate
        episode_metrics.pop("won_episode")

        return episode_metrics

    def stop(self) -> None:
        """Stop the logger."""
        self.logger.stop
