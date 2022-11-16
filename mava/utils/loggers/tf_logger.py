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

"""Utilities for logging to the terminal."""
import time
import warnings
from typing import Dict

import numpy as np
import tensorflow as tf
from acme.utils.loggers import base
from tensorflow import Tensor

from mava.utils.config_utils import flatten_dict


def format_key(key: str) -> str:
    """Internal function for formatting keys in Tensorboard format."""
    return key.title().replace("_", "")


def format_key_histograms(key: str) -> str:
    """Internal function for formatting keys in Tensorboard format."""
    return key.title().replace(":", "_") + "_hist"


class TFSummaryLogger(base.Logger):
    """Logs to a tf.summary created in a given logdir.

    If multiple TFSummaryLogger are created with the same logdir, results will be
    categorized by labels.
    """

    def __init__(self, logdir: str, label: str = "Logs"):
        """Initializes the logger.

        Args:
            logdir: directory to which we should log files.
            label: label string to use when logging. Default to 'Logs'.
        """
        self._time = time.time()
        self._label = label
        self._iter = 0
        self._logdir = logdir
        self._summary = tf.summary.create_file_writer(self._logdir)

    def write(self, values: base.LoggingData) -> None:
        """Write logging data.

        Args:
            values : values to write.
        """
        with self._summary.as_default():
            try:
                if isinstance(values, dict):
                    for key, value in values.items():
                        is_scalar_array = hasattr(value, "shape") and (
                            value.shape == [1] or value.shape == 1 or value.shape == ()
                        )
                        if np.isscalar(value) or is_scalar_array:
                            self.scalar_summary(key, value)
                        elif hasattr(value, "shape"):
                            self.histogram_summary(key, value)
                        elif isinstance(value, dict):
                            flattened_dict = flatten_dict(parent_key=key, d=value)
                            self.write(flattened_dict)
                        elif isinstance(value, tuple) or isinstance(value, list):
                            for index, elements in enumerate(value):
                                self.write({f"{key}_info_{index}": elements})
                        else:
                            warnings.warn(
                                f"Unable to log: {key}, unknown type: {type(value)}"
                            )
                elif isinstance(values, tuple) or isinstance(value, list):
                    for elements in values:
                        self.write(elements)
                else:
                    warnings.warn(
                        f"Unable to log: {values}, unknown type: {type(values)}"
                    )
            except Exception as ex:
                warnings.warn(
                    f"Unable to log: {key}, type: {type(value)} , value: {value}"
                    + f"ex: {ex}"
                )
            self._iter += 1

    def scalar_summary(self, key: str, value: float) -> None:
        """Log scalar.

        Args:
            key : key.
            value : value.
        """
        tf.summary.scalar(f"{self._label}/{format_key(key)}", value, step=self._iter)

    def dict_summary(self, key: str, value: Dict) -> None:
        """Log dict.

        Args:
            key : key.
            value : value.
        """
        dict_info = flatten_dict(parent_key=key, d=value)
        for (k, v) in dict_info.items():
            self.scalar_summary(k, v)

    def histogram_summary(self, key: str, value: Tensor) -> None:
        """Log histogram.

        Args:
            key : key.
            value : value.
        """
        tf.summary.histogram(
            f"{self._label}/{format_key_histograms(key)}", value, step=self._iter
        )

    def close(self) -> None:
        """Close logger."""
        self._summary.close()
