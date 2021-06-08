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

"""helper functions for logging"""

import abc
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional, Union

from acme.utils import loggers, paths
from acme.utils.loggers import base

from mava.utils.loggers.tf_logger import TFSummaryLogger


class MavaLogger(abc.ABC):
    @abc.abstractmethod
    def __init__(
        self,
        label: str,
        *args: Any,
        **kwargs: Any,
    ):
        """Init function that takes in a label and possibly other variables."""

    @abc.abstractmethod
    def write(self, data: Any) -> None:
        """Function that writes logged data."""


class Logger(MavaLogger):
    def __init__(
        self,
        label: str,
        directory: Union[Path, str],
        to_terminal: bool = True,
        to_csv: bool = False,
        to_tensorboard: bool = False,
        time_delta: float = 1.0,
        print_fn: Callable[[str], None] = print,
        time_stamp: Optional[str] = None,
        external_logger: Optional[base.Logger] = None,
        **external_logger_kwargs: Any,
    ):
        self._label = label

        if not isinstance(directory, Path):
            directory = Path(directory)

        self._directory = directory
        self._time_stamp = time_stamp if time_stamp else str(datetime.now())
        self._logger = self.make_logger(
            to_terminal,
            to_csv,
            to_tensorboard,
            time_delta,
            print_fn,
            external_logger=external_logger,
            **external_logger_kwargs,
        )
        self._logger_info = (
            to_terminal,
            to_csv,
            to_tensorboard,
            time_delta,
            print_fn,
            self._time_stamp,
        )

    def update_label(self, label: str) -> None:
        self._label = f"{self._label}_{label}"

    def make_logger(
        self,
        to_terminal: bool,
        to_csv: bool,
        to_tensorboard: bool,
        time_delta: float,
        print_fn: Callable[[str], None],
        external_logger: Optional[base.Logger],
        **external_logger_kwargs: Any,
    ) -> loggers.Logger:
        """Build a Mava logger.

        Args:
            label: Name to give to the logger.
            directory: base directory for the  logging of the experiment.
            to_terminal: to print the logs in the terminal.
            to_csv: to save the logs in a csv file.
            to_tensorboard: to write the logs tf-events.
            time_delta: minimum elapsed time (in seconds) between logging events.
            print_fn: function to call which acts like print.
            external_logger: optional external logger.
            external_logger_kwargs: optional external logger params.
        Returns:
            A logger (pipe) object that responds to logger.write(some_dict).
        """
        logger = []

        if to_terminal:
            logger += [loggers.TerminalLogger(label=self._label, print_fn=print_fn)]

        if to_csv:
            logger += [
                loggers.CSVLogger(
                    directory_or_file=self._path("csv"), label=self._label
                )
            ]

        if to_tensorboard:
            logger += [
                TFSummaryLogger(logdir=self._path("tensorboard"), label=self._label)
            ]

        if external_logger:
            logger += [
                external_logger(
                    label=self._label,
                    **external_logger_kwargs,
                )
            ]

        if logger:
            logger = loggers.Dispatcher(logger)
            logger = loggers.NoneFilter(logger)
            logger = loggers.TimeFilter(logger, time_delta)
        else:
            logger = loggers.NoOpLogger()

        return logger

    def _path(self, subdir: Optional[str] = None) -> str:
        if subdir:
            path = str(self._directory / self._time_stamp / subdir / self._label)
        else:
            path = str(self._directory / self._time_stamp / self._label)

        # Recursively replace "~"
        return paths.process_path(path)

    def write(self, data: Any) -> None:
        self._logger.write(data)
