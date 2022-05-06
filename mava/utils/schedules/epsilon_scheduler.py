from abc import ABC, abstractmethod


class EpsilonScheduler(ABC):
    """Abstract class for epsilon scheduler."""

    def __init__(self) -> None:
        """Initializes the epsilon scheduler with default value of 1.0."""
        self.current_epsilon: float = 1.0

    @abstractmethod
    def update(self, step: int) -> None:
        """Updates the value of self.current_epsilon based on the step number.

        Args:
            step: the step at which the interpolation is performed (commonly the
            current step).

        """
        pass

    @property
    def epsilon(self) -> float:
        """Returns the current epsilon."""
        return self.current_epsilon
