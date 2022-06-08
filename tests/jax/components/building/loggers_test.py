import functools
from typing import Callable

import pytest

from mava.components.jax.building.loggers import Logger, LoggerConfig
from mava.core_jax import SystemBuilder
from mava.systems.jax import Builder
from mava.utils.loggers import logger_utils


class TestLogger(Logger):
    """Test Logger component."""

    def __init__(
        self,
        test_logger_factory: Callable,
    ):
        """Create TestLogger

        Args:
            test_logger_factory: factory to use in the logger config.
        """
        logger_config = LoggerConfig()
        logger_config.logger_factory = test_logger_factory
        logger_config.logger_config = {
            "trainer": {"time_stamp": "trainer_config"},
            "executor": {"time_stamp": "executor_config"},
            "evaluator": {"time_stamp": "evaluator_config"},
        }

        super().__init__(logger_config)


@pytest.fixture
def test_logger_factory() -> Callable:
    """Pytest fixture for logger factory.

    Returns:
        Logger factory using default Mava logger.
    """
    simple_factory = functools.partial(
        logger_utils.make_logger,
        directory="~/mava",
        to_terminal=True,
        to_tensorboard=True,
        time_stamp="01/01/1997-00:00:00",
        time_delta=10,
    )

    return simple_factory


@pytest.fixture
def test_builder() -> SystemBuilder:
    """Pytest fixture for system builder. Added executor and trainer IDs to the store.

    Returns:
        System builder with no components.
    """
    system_builder = Builder(components=[])
    system_builder.store.executor_id = "executor_1"
    system_builder.store.trainer_id = "trainer_2"
    return system_builder


@pytest.fixture
def test_logger(test_logger_factory: Callable) -> Logger:
    """Pytest fixture for TestLogger.

    Args:
        test_logger_factory: factory to use in logger config.

    Returns:
        Default TestLogger.
    """
    test_logger = TestLogger(test_logger_factory)
    return test_logger


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
    assert test_builder.store.executor_logger._time_stamp == "executor_config"


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
    assert test_builder.store.executor_logger._time_stamp == "evaluator_config"


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
    assert test_builder.store.trainer_logger._time_stamp == "trainer_config"
