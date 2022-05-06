from mava.utils.schedules.epsilon_scheduler import EpsilonScheduler


class LinearEpsilonScheduler(EpsilonScheduler):
    """Linear Epsilon Scheduler.

    Notes:
        - Epsilon is changed linearly from initial_epsilon to final_epsilon over
          the course of the decay_steps.
        - After the decay_steps, the epsilon is equal to the final_epsilon.

    Args:
        start_value (float): start value
        end_value (float): end value
        steps (int): number of steps
    """

    def __init__(
        self, initial_epsilon: float, final_epsilon: float, decay_steps: float
    ) -> None:
        """Initializes the LinearEpsilonScheduler."""
        super().__init__()
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps
        self.current_epsilon = self.initial_epsilon

    def update(self, step: int) -> None:
        """Updates the epsilon, by interpolating between initial_epsilon and final_epsilon.

        Args:
            step: the step at which the interpolation is performed (commonly the
            current step).

        """
        self.current_epsilon = self.initial_epsilon + (
            self.final_epsilon - self.initial_epsilon
        ) * min(float(step) / self.decay_steps, 1)
