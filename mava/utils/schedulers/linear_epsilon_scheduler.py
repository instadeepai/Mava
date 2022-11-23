from mava.utils.schedulers.epsilon_scheduler import EpsilonScheduler


class LinearEpsilonScheduler(EpsilonScheduler):
    """Linear Epsilon Scheduler.

    Notes:
        - Epsilon is changed linearly from initial_epsilon to final_epsilon over
          the course of the decay_steps.
        - After the decay_steps, the epsilon is equal to the final_epsilon.
    """

    def __init__(
        self, initial_epsilon: float, final_epsilon: float, decay_steps: int
    ) -> None:
        """Initializes the LinearEpsilonScheduler.

        Args:
            initial_epsilon: start value for epsilon
            final_epsilon: end value for epsilon
            decay_steps: number of steps to move from initial_epsilon to final_epsilon
        """
        super().__init__()
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps

    def __call__(self, step: int) -> float:
        """Calculates the value for epsilon and returns it

        Interpolates between initial_epsilon and final_epsilon.

        Args:
            step: the step at which the interpolation is performed

        Returns:
            epsilon
        """
        return self.initial_epsilon + (self.final_epsilon - self.initial_epsilon) * min(
            float(step) / self.decay_steps, 1
        )
