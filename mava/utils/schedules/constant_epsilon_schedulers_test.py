from mava.utils.schedules.constant_epsilon_scheduler import ConstantEpsilonScheduler


def test_constant_epsilon_scheduler_init() -> None:
    """Test that the LinearEpsilonScheduler initializes correctly."""
    epsilon_scheduler = ConstantEpsilonScheduler(0.1)
    assert epsilon_scheduler.epsilon == 0.1


def test_linear_epsilon_scheduler_update() -> None:
    """Test that the LinearEpsilonScheduler updates correctly."""
    epsilon_scheduler = ConstantEpsilonScheduler(0.1)
    epsilon_scheduler.update(0)
    assert epsilon_scheduler.epsilon == 0.1
    epsilon_scheduler.update(50)
    assert epsilon_scheduler.epsilon == 0.1
