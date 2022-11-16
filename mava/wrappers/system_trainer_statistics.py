# python3
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

"""Generic environment loop wrapper to track system statistics"""

import time
from typing import Any, Dict, List, Sequence

import numpy as np
from acme.utils import loggers

import mava
from mava.utils import training_utils as train_utils
from mava.utils.loggers import Logger
from mava.utils.wrapper_utils import RunningStatistics


class TrainerWrapperBase(mava.Trainer):
    def __init__(
        self,
        trainer: mava.Trainer,
    ) -> None:
        """Init for trainer wrapper.

        Args:
            trainer : base trainer class.
        """
        self._trainer = trainer

    def get_variables(self, names: Sequence[str]) -> Dict[str, Dict[str, np.ndarray]]:
        """Get vars for trainer.

        Args:
            names : names of vars.

        Returns:
            trainer vars.
        """
        return self._trainer.get_variables(names)

    def _create_loggers(self, keys: List[str]) -> None:
        """Creates logger.

        Args:
            keys : keys used by logger.

        Raises:
            NotImplementedError: currently not implemented.
        """
        raise NotImplementedError

    def __getattr__(self, name: str) -> Any:
        """Expose any other attributes of the underlying trainer."""
        return getattr(self._trainer, name)


class TrainerStatisticsBase(TrainerWrapperBase):
    def __init__(
        self,
        trainer: mava.Trainer,
    ) -> None:
        """Init for trainer stats base class.

        Args:
            trainer : internal trainer.
        """
        super().__init__(trainer)
        self._require_loggers = True

    def step(self) -> None:
        """Trainer step."""
        # Run the learning step.
        fetches = self._step()

        if self._require_loggers:
            self._create_loggers(list(fetches.keys()))
            self._require_loggers = False

        # compute statistics
        self._compute_statistics(fetches)

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp: float = timestamp

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        fetches.update(counts)

        if self._system_checkpointer:
            train_utils.checkpoint_networks(self._system_checkpointer)

        if self._logger:
            self._logger.write(fetches)

    def after_trainer_step(self) -> None:
        """Function called after trainer step."""
        if hasattr(self._trainer, "after_trainer_step"):
            self._trainer.after_trainer_step()


class DetailedTrainerStatistics(TrainerStatisticsBase):
    """A trainer class that logs episode stats."""

    def __init__(
        self,
        trainer: mava.Trainer,
        metrics: List[str] = ["policy_loss"],
        summary_stats: List = ["mean", "max", "min", "var", "std"],
    ) -> None:
        """Init for detailed trainer stats class.

        Args:
            trainer : internal trainer.
            metrics : metrics.
            summary_stats : what aggregations to apply to metrics.
        """
        super().__init__(trainer)

        self._metrics = metrics
        self._summary_stats = summary_stats

    def _create_loggers(self, keys: List[str]) -> None:
        """Func to create loggers.

        Args:
            keys : key for loggers.
        """

        # get system logger data
        trainer_label = self._logger._label
        base_dir = self._logger._directory
        (
            to_terminal,
            to_csv,
            to_tensorboard,
            time_delta,
            print_fn,
            time_stamp,
        ) = self._logger._logger_info

        self._network_running_statistics: Dict[str, Dict[str, float]] = {}
        self._networks_stats: Dict[str, Dict[str, RunningStatistics]] = {
            key: {} for key in keys
        }
        self._network_loggers: Dict[str, loggers.Logger] = {}

        # statistics dictionary
        for key in keys:
            network_label = trainer_label + "_" + key
            self._network_loggers[key] = Logger(
                label=network_label,
                directory=base_dir,
                to_terminal=to_terminal,
                to_csv=to_csv,
                to_tensorboard=to_tensorboard,
                time_delta=0,
                print_fn=print_fn,
                time_stamp=time_stamp,
            )
            for metric in self._metrics:
                self._networks_stats[key][metric] = RunningStatistics(f"{key}_{metric}")

    def _compute_statistics(self, data: Dict[str, Dict[str, float]]) -> None:
        """Compute stats for trainer.

        Args:
            data : trainer stats data.
        """
        for network, datum in data.items():
            for key, val in datum.items():
                network_running_statistics: Dict[str, float] = {}
                network_running_statistics[f"{network}_raw_{key}"] = val
                if key in self._networks_stats[network]:
                    self._networks_stats[network][key].push(val)
                    for stat in self._summary_stats:
                        network_running_statistics[
                            f"{network}_{stat}_{key}"
                        ] = self._networks_stats[network][key].__getattribute__(stat)()

                self._network_loggers[network].write(network_running_statistics)


class ScaledTrainerStatisticsBase(TrainerWrapperBase):
    def __init__(
        self,
        trainer: mava.Trainer,
    ) -> None:
        """Init for trainer stats base class.

        Args:
            trainer : internal trainer.
        """
        super().__init__(trainer)
        self._require_loggers = True

    def step(self) -> None:
        """Trainer step."""
        # Run the learning step.
        fetches = self._step()
        if self._require_loggers:
            self._create_loggers(list(fetches.keys()))
            self._require_loggers = False

        # compute statistics
        self._compute_statistics(fetches)

        # Compute elapsed time.
        timestamp = time.time()
        elapsed_time = timestamp - self._timestamp if self._timestamp else 0
        self._timestamp: float = timestamp

        # Update our counts and record it.
        self._variable_client.add_async(
            ["trainer_steps", "trainer_walltime"],
            {"trainer_steps": 1, "trainer_walltime": elapsed_time},
        )

        # Set and get the latest variables
        self._variable_client.set_and_get_async()

        fetches.update(self._counts)

        if self._logger:
            self._logger.write(fetches)


class ScaledDetailedTrainerStatistics(ScaledTrainerStatisticsBase):
    """A trainer class that logs episode stats."""

    def __init__(
        self,
        trainer: mava.Trainer,
        metrics: List[str] = ["policy_loss"],
        summary_stats: List = ["mean", "max", "min", "var", "std"],
    ) -> None:
        """Init for detailed trainer stats class.

        Args:
            trainer : internal trainer.
            metrics : metrics.
            summary_stats : what aggregations to apply to metrics.
        """
        super().__init__(trainer)

        self._metrics = metrics
        self._summary_stats = summary_stats

    def _create_loggers(self, keys: List[str]) -> None:
        """Func to create loggers.

        Args:
            keys : key for loggers.
        """
        # get system logger data
        trainer_label = self._logger._label
        base_dir = self._logger._directory
        (
            to_terminal,
            to_csv,
            to_tensorboard,
            time_delta,
            print_fn,
            time_stamp,
        ) = self._logger._logger_info

        self._network_running_statistics: Dict[str, Dict[str, float]] = {}
        self._networks_stats: Dict[str, Dict[str, RunningStatistics]] = {
            key: {} for key in keys
        }
        self._network_loggers: Dict[str, loggers.Logger] = {}

        # statistics dictionary
        for key in keys:
            network_label = trainer_label + "_" + key
            self._network_loggers[key] = Logger(
                label=network_label,
                directory=base_dir,
                to_terminal=to_terminal,
                to_csv=to_csv,
                to_tensorboard=to_tensorboard,
                time_delta=0,
                print_fn=print_fn,
                time_stamp=time_stamp,
            )
            for metric in self._metrics:
                self._networks_stats[key][metric] = RunningStatistics(f"{key}_{metric}")

    def _compute_statistics(self, data: Dict[str, Dict[str, float]]) -> None:
        """Compute stats for trainer.

        Args:
            data : trainer stats data.
        """
        for network, datum in data.items():
            for key, val in datum.items():
                network_running_statistics: Dict[str, float] = {}
                network_running_statistics[f"{network}_raw_{key}"] = val
                self._networks_stats[network][key].push(val)
                for stat in self._summary_stats:
                    network_running_statistics[
                        f"{network}_{stat}_{key}"
                    ] = self._networks_stats[network][key].__getattribute__(stat)()

                self._network_loggers[network].write(network_running_statistics)
