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
import tempfile
from typing import Dict

import jax.numpy as jnp
import pytest

from mava.utils.loggers.eval_json_logger import JSONLogger

temp_path = tempfile.mkdtemp()


@pytest.fixture
def test_data() -> Dict:
    """Logger arguments."""

    return {
        "experiment_path": temp_path,
        "random_seed": 1111,
        "env_name": "test_env",
        "task_name": "test_task",
        "system_name": "test_system",
    }


@pytest.fixture
def logger(test_data: Dict) -> JSONLogger:
    """Logger object for testing."""

    logger = JSONLogger(
        experiment_path=test_data["experiment_path"],
        random_seed=test_data["random_seed"],
        env_name=test_data["env_name"],
        task_name=test_data["task_name"],
        system_name=test_data["system_name"],
    )
    return logger


@pytest.fixture
def jax_type_step_data() -> Dict:
    """Mock data for a logging step in jax format."""
    return {
        "mock_normal_step_data": {
            "step_count": jnp.array([10000]),
            "metric_1": jnp.array([1, 1, 1, 1, 1]),
            "metric_2": jnp.array([2]),
        },
        "mock_absolute_metric_data": {
            "metric_1": jnp.array([11, 11, 11, 11, 11]),
            "metric_2": jnp.array([22]),
        },
    }


@pytest.fixture
def python_type_step_data() -> Dict:
    """Mock data for a logging step in python native format."""

    return {
        "mock_normal_step_data": {
            "step_count": [10000],
            "metric_1": [1, 1, 1, 1, 1],
            "metric_2": [2],
        },
        "mock_absolute_metric_data": {
            "metric_1": [11, 11, 11, 11, 11],
            "metric_2": [22],
        },
    }


def test_logger_init(test_data: Dict, logger: JSONLogger) -> None:
    """Test that json logger initialises correctly"""

    # Verify path and variable initialisation
    assert logger._log_dir == str(temp_path) + "/json_data/test_env/test_task/"
    assert (
        logger._logs_file_dir
        == str(temp_path)
        + "/json_data/test_env/test_task/"
        + "test_env_test_task_run1111_evaluation_data.json"
    )
    assert logger._step_count == 0
    assert logger._random_seed == str(test_data["random_seed"])
    assert logger._env_name == test_data["env_name"]
    assert logger._task_name == test_data["task_name"]
    assert logger._system_name == test_data["system_name"]

    # Verify json file structure
    with open(logger._logs_file_dir, "r") as f:
        read_in_data = json.load(f)

    assert read_in_data == {
        test_data["env_name"]: {
            test_data["task_name"]: {
                test_data["system_name"]: {str(test_data["random_seed"]): {}}
            }
        }
    }


def test_jsonify_and_process(
    jax_type_step_data: Dict,
    logger: JSONLogger,
) -> None:
    """Test that data is correctly converted to python native types."""

    mock_normal_step_data = jax_type_step_data["mock_normal_step_data"]

    logger._jsonify_and_process(mock_normal_step_data)

    assert logger._results_dict == {
        "step_count": [10000],
        "metric_1": [1, 1, 1, 1, 1],
        "metric_2": [2],
    }


def test_add_data_to_dictionary(
    test_data: Dict, python_type_step_data: Dict, logger: JSONLogger
) -> None:
    """Test that add_data_to_dictionary method works."""

    mock_original_dict: Dict = {
        test_data["env_name"]: {
            test_data["task_name"]: {
                test_data["system_name"]: {str(test_data["random_seed"]): {}}
            }
        }
    }

    mock_normal_step_data = python_type_step_data["mock_normal_step_data"]
    mock_absolute_metric_data = python_type_step_data["mock_absolute_metric_data"]

    # write one normal step
    assert logger._add_data_to_dictionary(
        mock_original_dict, mock_normal_step_data
    ) == {
        "test_env": {
            "test_task": {
                "test_system": {
                    "1111": {
                        "step_0": {
                            "step_count": [10000],
                            "metric_1": [1, 1, 1, 1, 1],
                            "metric_2": [2],
                        }
                    }
                }
            }
        }
    }

    # Write one absolute metric step
    assert logger._add_data_to_dictionary(
        mock_original_dict, mock_absolute_metric_data
    ) == {
        "test_env": {
            "test_task": {
                "test_system": {
                    "1111": {
                        "step_0": {
                            "step_count": [10000],
                            "metric_1": [1, 1, 1, 1, 1],
                            "metric_2": [2],
                        },
                        "absolute_metrics": {
                            "metric_1": [11, 11, 11, 11, 11],
                            "metric_2": [22],
                        },
                    }
                }
            }
        }
    }
