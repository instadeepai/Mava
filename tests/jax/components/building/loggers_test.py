from typing import Any, Callable, Tuple

import pytest

from mava.components.jax.building.loggers import Logger, LoggerConfig
from mava.core_jax import SystemBuilder
from mava.systems.jax import Builder


class TestLogger(Logger):
    def __init__(
        self,
        test_logger_factory: Callable,
    ):
        logger_config = LoggerConfig()
        logger_config.logger_factory = test_logger_factory
        logger_config.logger_config = {
            "trainer": LoggerConfig(logger_config="trainer_config"),
            "executor": LoggerConfig(logger_config="executor_config"),
            "evaluator": LoggerConfig(logger_config="evaluator_config"),
        }

        super().__init__(logger_config)

        # self.on_building_executor_logger(builder)
        # self.on_building_trainer_logger(builder)


@pytest.fixture
def test_logger_factory() -> Callable:
    def simple_factory(component_id: int, config: LoggerConfig) -> Tuple[int, Any]:
        return component_id, config.logger_config

    return simple_factory


@pytest.fixture
def test_builder() -> SystemBuilder:
    system_builder = Builder(components=[])
    system_builder.store.executor_id = 1
    system_builder.store.trainer_id = 2
    return system_builder


@pytest.fixture
def test_executor_logger(
    test_builder: SystemBuilder, test_logger_factory: Callable
) -> Logger:
    test_builder.store.is_evaluator = False

    test_logger = TestLogger(test_logger_factory)
    return test_logger


def test_assert_true(test_executor_logger: Logger) -> None:
    assert test_executor_logger.name() == "logger"
    print(test_executor_logger.config.logger_config)
    assert True
