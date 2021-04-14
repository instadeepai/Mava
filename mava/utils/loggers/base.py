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

from pathlib import Path
from typing import Any, Callable, Optional

from acme.utils import loggers

from mava.utils.loggers.tf_logger import TFSummaryLogger


def path(log_dir: str, subdir: Optional[str] = None) -> str:
    return log_dir + "/" + subdir if subdir else log_dir


def make_logger(
    label: str,
    directory: Path,
    to_terminal: bool = True,
    to_csv: bool = False,
    to_tensorboard: bool = False,
    time_delta: float = 1.0,
    print_fn: Callable[[str], None] = print,
    **kwargs: Any,
) -> loggers.Logger:
    """Build an Acme logger.

    Args:
        label: Name to give to the logger.
        directory: base directory for the  logging of the experiment.
        to_terminal: to print the logs in the terminal.
        to_csv: to save the logs in a csv file.
        to_tensorboard: to write the logs tf-events.
        to_wandb: whether to use wandb.
        to_neptune: whether to use neptune.
        time_delta: minimum elapsed time (in seconds) between logging events.
        print_fn: function to call which acts like print.

    Returns:
        A logger (pipe) object that responds to logger.write(some_dict).
    """
    logger = []
    # path = lambda d: str(directory / d) if d else str(directory)

    if to_terminal:
        logger += [loggers.TerminalLogger(label=label, print_fn=print_fn)]

    if to_csv:
        logger += [
            loggers.CSVLogger(directory_or_file=path("csv"), label=label)
        ]  # type: ignore

    if to_tensorboard:
        logger += [
            TFSummaryLogger(logdir=path("tensorboard"), label=label)
        ]  # type: ignore

    if logger:
        logger = loggers.Dispatcher(logger)
        logger = loggers.NoneFilter(logger)
        logger = loggers.TimeFilter(logger, time_delta)
    else:
        logger = loggers.NoOpLogger()

    return logger
