import numpy as np
import pytest

from mava.utils.schedulers.linear_epsilon_scheduler import LinearEpsilonScheduler


@pytest.fixture
def scheduler() -> LinearEpsilonScheduler:
    """Returns a LinearEpsilonScheduler"""
    return LinearEpsilonScheduler(1.0, 0.0, 10)


def test___init__(scheduler: LinearEpsilonScheduler) -> None:
    """Tests epsilon scheduler init"""
    assert scheduler.initial_epsilon == 1.0
    assert scheduler.final_epsilon == 0.0
    assert scheduler.decay_steps == 10


def test___call__(scheduler: LinearEpsilonScheduler) -> None:
    """Checks that epsilons are generated as is expected"""
    epsilons = reversed(np.arange(11) / 10)
    for step, epsilon in enumerate(epsilons):
        np.testing.assert_almost_equal(scheduler(step), epsilon)
