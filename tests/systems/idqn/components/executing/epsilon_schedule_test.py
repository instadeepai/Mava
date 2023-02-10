from types import SimpleNamespace

import pytest

from mava.systems.executor import Executor
from mava.systems.idqn.components.executing.epsilon_scheduler import (
    EpsilonScheduler,
    EpsilonSchedulerConfig,
)
from mava.utils.schedulers.linear_epsilon_scheduler import LinearEpsilonScheduler


@pytest.fixture
def executor() -> Executor:
    """Creates executor"""
    return Executor(SimpleNamespace(is_evaluator=False), [])


def test_on_execution_init_end_no_config(executor: Executor) -> None:
    """Checks that epsilon scheduler is created"""
    EpsilonScheduler().on_execution_init_end(executor)

    assert isinstance(executor.store.epsilon_scheduler, LinearEpsilonScheduler)
    assert executor.store.epsilon_scheduler.initial_epsilon == 1.0
    assert executor.store.epsilon_scheduler.final_epsilon == 0.1
    assert executor.store.epsilon_scheduler.decay_steps == 10_000


def test_on_execution_init_end_config(executor: Executor) -> None:
    """Checks that epsilon scheduler is created"""
    schedule = lambda x: x + 1  # noqa
    EpsilonScheduler(EpsilonSchedulerConfig(schedule)).on_execution_init_end(executor)

    assert executor.store.epsilon_scheduler == schedule
