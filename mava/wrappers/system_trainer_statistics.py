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

"""Generic environment loop wrapper to track system statistics"""

import time
from typing import Any, Dict, List

from acme.utils import loggers

import mava
from mava.utils.loggers import Logger
from mava.utils.wrapper_utils import RunningStatistics


class TrainerStatisticsBase:
    """A parallel MARL environment loop.
    This takes `Environment` and `Executor` instances and coordinates their
    interaction. Executors are updated if `should_update=True`. This can be used as:
        loop = EnvironmentLoop(environment, executor)
        loop.run(num_episodes)
    A `Counter` instance can optionally be given in order to maintain counts
    between different Mava components. If not given a local Counter will be
    created to maintain counts between calls to the `run` method.
    A `Logger` instance can also be passed in order to control the output of the
    loop. If not given a platform-specific default logger will be used as defined
    by utils.loggers.make_default_logger from acme. A string `label` can be passed
    to easily change the label associated with the default logger; this is ignored
    if a `Logger` instance is given.
    """

    def __init__(
        self,
        trainer: mava.Trainer,
    ) -> None:
        self._trainer = trainer
        self._require_loggers = True

        # NOTE (Arnu): if I try inheriting from mava.Trainer
        # I get the following error:
        # "TypeError: Can't instantiate abstract class ...
        # DetailedTrainerStatistics with abstract methods get_variables"
        # If I hardcode the type here it breaks the logging for some reason
        # For now as is, there is a type mismatch in the system code when
        # returning the wrapped trainer, but everything seems to work.
        # Need to find a solution to this.
        # self.__class__ = mava.Trainer

    def _create_loggers(self, keys: List[str]) -> None:
        raise NotImplementedError

    def _compute_statistics(self, data: Dict[str, Dict[str, float]]) -> None:
        raise NotImplementedError

    def __getattr__(self, attr: Any) -> Any:
        return self._trainer.__getattribute__(attr)

    def step(self) -> None:
        # Run the learning step.
        fetches = self._step()

        if self._require_loggers:
            self._create_loggers(list(fetches.keys()))
            self._require_loggers = False

        # compute statistics
        self._compute_statistics(fetches)

        # Compute elapsed time.
        # NOTE (Arnu): getting type issues with the timestamp
        # not sure why. Look into a fix for this.
        timestamp = time.time()
        if self._timestamp:  # type: ignore
            elapsed_time = timestamp - self._timestamp  # type: ignore
        else:
            elapsed_time = 0
        self._timestamp = timestamp  # type: ignore

        # Update our counts and record it.
        counts = self._counter.increment(steps=1, walltime=elapsed_time)
        # fetches.update(counts)

        # Checkpoint the networks.
        if len(self._system_checkpointer.keys()) > 0:
            for network_key in self.unique_net_keys:
                checkpointer = self._system_checkpointer[network_key]
                checkpointer.save()

        self._logger.write(counts)


class DetailedTrainerStatistics(TrainerStatisticsBase):
    def __init__(
        self, trainer: mava.Trainer, metrics: List[str] = ["policy_loss"]
    ) -> None:
        super().__init__(trainer)

        self._metrics = metrics

    def _create_loggers(self, keys: List[str]) -> None:

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
        self._summary_stats = ["mean", "max", "min", "var", "std"]

        for network, datum in data.items():
            for key, val in datum.items():
                network_running_statistics: Dict[str, float] = {}
                self._networks_stats[network][key].push(val)
                for stat in self._summary_stats:
                    network_running_statistics[
                        f"{network}_{stat}_{key}"
                    ] = self._networks_stats[network][key].__getattribute__(stat)()
                self._network_loggers[network].write(network_running_statistics)
