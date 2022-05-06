from mava.utils.schedules.linear_epsilon_scheduler import LinearEpsilonScheduler


def test_linear_epsilon_scheduler_init() -> None:
    """Test that the LinearEpsilonScheduler initializes correctly."""
    epsilon_scheduler = LinearEpsilonScheduler(0.1, 0.9, 100)
    assert epsilon_scheduler.epsilon == 0.1


def test_linear_epsilon_scheduler_update() -> None:
    """Test that the LinearEpsilonScheduler updates correctly."""
    epsilon_scheduler = LinearEpsilonScheduler(0.1, 0.9, 100)
    epsilon_scheduler.update(0)
    assert epsilon_scheduler.epsilon == 0.1
    epsilon_scheduler.update(50)
    assert epsilon_scheduler.epsilon == 50.0 / 100 * (0.9 + 0.1)
    epsilon_scheduler.update(100)
    assert epsilon_scheduler.epsilon == 0.9
    epsilon_scheduler.update(150)
    assert epsilon_scheduler.epsilon == 0.9
