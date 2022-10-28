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

import json
import os
from typing import Any, Dict

import jax
from acme.utils import paths


class JSONLogger:
    def __init__(
        self,
        experiment_path: str,
        random_seed: int,
        env_name: str,
        task_name: str,
        system_name: str,
    ) -> None:
        """Initialise JSON logger

        Args:
            experiment_path: path where experiment data should be logged
            random_seed: random seed used for the experiment
            env_name: name of environment of experiment. eg. "SMAC"
            task_name: name of current experiment task. eg. "3m"
            system_name: name of system being evaluated. eg. "IPPO"
        """

        # Create subfolder for storing json file
        self._log_dir = experiment_path + f"/json_data/{env_name}/{task_name}/"
        self._log_dir = paths.process_path(self._log_dir, add_uid=False)

        self._logs_file_dir = (
            f"{self._log_dir}{env_name}_{task_name}"
            + f"_run{str(random_seed)}_evaluation_data.json"
        )

        self._step_count = 0
        self._random_seed = str(random_seed)
        self._env_name = env_name
        self._task_name = task_name
        self._system_name = system_name

        # If directory doesn't exist create it
        if not os.path.exists(self._log_dir):
            os.makedirs(self._log_dir)

        # Initialise json file for logging data
        if not os.path.exists(self._logs_file_dir):
            with open(self._logs_file_dir, "w+") as f:
                json.dump(
                    {
                        self._env_name: {
                            self._task_name: {
                                self._system_name: {self._random_seed: {}}
                            }
                        }
                    },
                    f,
                    indent=4,
                )

    def _jsonify_and_process(self, results_dict: Dict[str, Any]) -> None:
        """Convert all elements to be logged to native python types."""

        # convert all leaves to python types by calling tolist()
        results_dict = jax.tree_util.tree_map(lambda leaf: leaf.tolist(), results_dict)

        self._results_dict = results_dict

    def _add_data_to_dictionary(
        self, original_dictionary: Dict[str, Any], dictionary_to_add: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Adds new data to already logged data read in from a json file."""
        if "step_count" not in list(dictionary_to_add.keys()):
            original_dictionary[self._env_name][self._task_name][self._system_name][
                self._random_seed
            ]["absolute_metrics"] = dictionary_to_add
        else:
            original_dictionary[self._env_name][self._task_name][self._system_name][
                self._random_seed
            ][f"step_{str(self._step_count)}"] = dictionary_to_add
        return original_dictionary

    def write(self, results_dict: Dict[str, jax.numpy.ndarray]) -> None:
        """Write current evaluation data to a json file.

        It should be noted that the input data here should be in one of two
        forms depending on whether evaluation metrics for a particular
        evaluation step are being logged or whether absolute metrics are
        being logged.

        For evaluation metrics at a particular evaluation step, the
        `results_dict` should take the following form:
        results_dict = {
            'step_count': <value>,
            'metric_1': <array>,
            'metric_2': <array>
        }

        Where for absolute metrics, the `step_count` key should not be
        included, implying that the `results_dict` dictionary will look
        as follows:

        results_dict = {
            'metric_1': <array>,
            'metric_2': <array>
        }
        """

        self._jsonify_and_process(results_dict=results_dict)

        # Load current logged data
        with open(self._logs_file_dir, "r") as f:
            read_in_data = json.load(f)

        with open(self._logs_file_dir, "w+") as f:
            updated_data = self._add_data_to_dictionary(
                read_in_data, self._results_dict
            )
            json.dump(updated_data, f, indent=4)

        self._step_count += 1
