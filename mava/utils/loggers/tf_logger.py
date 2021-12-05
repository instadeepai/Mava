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

"""Utilities for logging to the terminal."""
import time
import warnings
from typing import Dict, List

import numpy as np
import tensorflow as tf
from acme.utils.loggers import base
from tensorflow import Tensor


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
                            flatten_dict = self._flatten_dict(
                                parent_key=key, dict_info=value
                            )
                            self.write(flatten_dict)
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
        tf.summary.scalar(f"{self._label}/{format_key(key)}", value, step=self._iter)

    def dict_summary(self, key: str, value: Dict) -> None:
        dict_info = self._flatten_dict(parent_key=key, dict_info=value)
        for (k, v) in dict_info.items():
            self.scalar_summary(k, v)

    def histogram_summary(self, key: str, value: Tensor) -> None:
        tf.summary.histogram(
            f"{self._label}/{format_key_histograms(key)}", value, step=self._iter
        )

    # Flatten dict, adapted from
    # https://stackoverflow.com/questions/6027558/flatten-nested-dictionaries-compressing-keys
    # Converts {'agent_0': {'critic_loss': 0.1, 'policy_loss': 0.2},...}
    #   to  {'agent_0_critic_loss':0.1,'agent_0_policy_loss':0.1 ,...}
    def _flatten_dict(
        self, parent_key: str, dict_info: Dict, sep: str = "_"
    ) -> Dict[str, float]:
        items: List = []
        for k, v in dict_info.items():
            k = str(k)
            new_key = parent_key + sep + k if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    self._flatten_dict(parent_key=new_key, dict_info=v, sep=sep).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)

    def close(self) -> None:
        self._summary.close()
