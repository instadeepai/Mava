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

"""Logger unit tests"""

import functools
import tempfile
from typing import Callable

import pytest

from mava.components.building.loggers import Logger, LoggerConfig
from mava.core_jax import SystemBuilder
from mava.systems import Builder
from mava.utils.loggers import logger_utils


@pytest.fixture
def test_logger_factory() -> Callable:
    """Pytest fixture for logger factory.

    Returns:
        Logger factory using default Mava logger.
    """
    simple_factory = functools.partial(
        logger_utils.make_logger,
        directory=tempfile.mkdtemp(),
        to_terminal=True,
        to_tensorboard=True,
        time_stamp="01/01/1997-00:00:00",
        time_delta=10,
    )

    return simple_factory


@pytest.fixture
def test_builder() -> SystemBuilder:
    """Pytest fixture for system builder. Adds executor and trainer IDs to the store.

    Returns:
        System builder with no components.
    """
    system_builder = Builder(components=[])
    system_builder.store.executor_id = "executor_1"
    system_builder.store.trainer_id = "trainer_2"
    system_builder.store.global_config.evaluation_interval = None
    system_builder.store.global_config.environment_factory = lambda evaluation: (
        (),
        {
            "environment_name": "test_env",
            "task_name": "test_task",
        },
    )
    system_builder.store.global_config.seed = 1111
    system_builder.store.global_config.system_name = "test_system"
    system_builder.store.global_config.json_path = None
    return system_builder


@pytest.fixture
def test_logger(test_logger_factory: Callable) -> Logger:
    """Pytest fixture for TestLogger.

    Args:
        test_logger_factory: factory to use in logger config.

    Returns:
        Default TestLogger.
    """
    logger_config = LoggerConfig()
    logger_config.logger_factory = test_logger_factory
    logger_config.logger_config = {
        "trainer": {"time_stamp": "trainer_logger_config"},
        "executor": {"time_stamp": "executor_logger_config"},
        "evaluator": {"time_stamp": "evaluator_logger_config"},
    }

    return Logger(logger_config)


@pytest.fixture
def test_logger_with_json(test_logger_factory: Callable) -> Logger:
    """Pytest fixture for TestLogger with json logging.

    Args:
        test_logger_factory: factory to use in logger config.

    Returns:
        Default TestLogger.
    """
    logger_config = LoggerConfig()
    logger_config.logger_factory = test_logger_factory
    logger_config.logger_config = {
        "trainer": {"time_stamp": "trainer_logger_config"},
        "executor": {"time_stamp": "executor_logger_config"},
        "evaluator": {
            "time_stamp": "evaluator_logger_config",
            "to_json": True,
        },
    }

    return Logger(logger_config)


@pytest.fixture
def test_s3_logger() -> Logger:
    """Pytest fixture for s3 logger (not a factory).

    Returns:
        Logger.
    """
    s3_logger = logger_utils.make_logger(
        directory="s3://test-url-path.com",
        to_terminal=True,
        to_tensorboard=False,
        time_stamp="01/01/1997-00:00:00",
        time_delta=10,
        label="s3-logger",
    )

    return s3_logger


def test_on_building_executor_logger_executor(
    test_logger: Logger, test_builder: SystemBuilder
) -> None:
    """Test on_building_executor_logger_executor method for executor.

    Args:
        test_logger: Fixture Logger.
        test_builder: Fixture SystemBuilder.

    Returns:
        None.
    """
    test_builder.store.is_evaluator = False
    test_logger.on_building_executor_logger(test_builder)

    # Correct component name
    assert test_logger.name() == "logger"

    # Correct logger has been created
    assert test_builder.store.executor_logger is not None
    assert not hasattr(test_builder.store, "trainer_logger")

    # Correct logger config has been loaded
    assert test_builder.store.executor_logger._label == "executor_1"
    assert test_builder.store.executor_logger._time_stamp == "executor_logger_config"


def test_on_building_executor_logger_evaluator(
    test_logger: Logger, test_builder: SystemBuilder
) -> None:
    """Test on_building_executor_logger_executor method for evaluator.

    Args:
        test_logger: Fixture Logger.
        test_builder: Fixture SystemBuilder.

    Returns:
        None.
    """
    test_builder.store.is_evaluator = True
    test_logger.on_building_executor_logger(test_builder)

    # Correct component name
    assert test_logger.name() == "logger"

    # Correct logger has been created
    assert test_builder.store.executor_logger is not None
    assert not hasattr(test_builder.store, "trainer_logger")

    # Correct logger config has been loaded
    assert test_builder.store.executor_logger._label == "executor_1"
    assert test_builder.store.executor_logger._time_stamp == "evaluator_logger_config"


def test_on_building_executor_logger_evaluator_with_json(
    test_logger_with_json: Logger, test_builder: SystemBuilder
) -> None:
    """Test whether json logger is correctly added to evaluator"""

    test_builder.store.is_evaluator = True
    test_logger_with_json.on_building_executor_logger(test_builder)

    # Correct component name
    assert test_logger_with_json.name() == "logger"

    # Correct logger has been created
    assert test_builder.store.executor_logger is not None
    assert not hasattr(test_builder.store, "trainer_logger")

    # Correct logger config has been loaded
    assert test_builder.store.executor_logger._label == "executor_1"
    assert test_builder.store.executor_logger._time_stamp == "evaluator_logger_config"


def test_on_building_trainer_logger(
    test_logger: Logger, test_builder: SystemBuilder
) -> None:
    """Test on_building_trainer_logger method for trainer.

    Args:
        test_logger: Fixture Logger.
        test_builder: Fixture SystemBuilder.

    Returns:
        None.
    """
    test_logger.on_building_trainer_logger(test_builder)

    # Correct component name
    assert test_logger.name() == "logger"

    # Correct logger has been created
    assert test_builder.store.trainer_logger is not None
    assert not hasattr(test_builder.store, "executor_logger")

    # Correct logger config has been loaded
    assert test_builder.store.trainer_logger._label == "trainer_2"
    assert test_builder.store.trainer_logger._time_stamp == "trainer_logger_config"


def test_logger_s3_url(test_s3_logger: Logger):
    """Test that correct paths are set when using s3 logger.

    Args:
        test_s3_logger : logger with s3 path.
    """
    assert test_s3_logger._directory == "s3://test-url-path.com"
    assert (
        test_s3_logger._path() == "s3://test-url-path.com/01/01/1997-00:00:00/s3-logger"
    )
