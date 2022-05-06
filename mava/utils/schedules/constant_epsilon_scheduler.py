from mava.utils.schedules.epsilon_scheduler import EpsilonScheduler


class ConstantEpsilonScheduler(EpsilonScheduler):
    """A constant Epsilon Scheduler.

    Args:
        value: The value to use for the constant epsilon.
    """

    def __init__(self, value: float):
        """Initializes the ConstantEpsilonScheduler."""
        super().__init__()
        self.current_epsilon = value

    def update(self, step: int) -> None:
        """No update is required for this scheduler."""
        pass

    @property
    def epsilon(self) -> float:
        """Returns the current epsilon."""
        return self.current_epsilon
