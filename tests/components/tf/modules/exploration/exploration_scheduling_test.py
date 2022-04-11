from mava.components.tf.modules.exploration.exploration_scheduling import (
    PieceWiseLinearTimestepScheduler as PWLTS,
)


class TestExplorationScheduling:
    """Testing the exploration schedule."""

    @staticmethod
    def create_piecewise_linear_exploration_schedule() -> PWLTS:
        """Create a piecewise linear exploration schedule for further tests."""

        exploration_schedule = PWLTS(
            timesteps=[50, 100, 200],
            epsilons=[0.3, 0.2, 0.6],
            initial_fallback_epsilon=0.9,
            final_fallback_epsilon=0.01,
        )

        return exploration_schedule

    def test_piecewise_linear_initial_epsilon(self) -> None:
        """Test the initial epsilon value."""
        exploration_schedule = self.create_piecewise_linear_exploration_schedule()

        # As long as the epsilon is not updated it should return the first element of
        # the epsilon list.
        assert exploration_schedule.get_epsilon() == 0.3

    def test_piecewise_linear_epsilon_update(self) -> None:
        """Test the epsilon update."""
        # Update the epsilon to a timestep before the first element of timestep list.
        # In this case it should return the initial fallback epsilon.
        exploration_schedule = self.create_piecewise_linear_exploration_schedule()

        for timestep in range(0, 50):
            exploration_schedule.decrement_epsilon(timestep)
            assert exploration_schedule.get_epsilon() == 0.9

        # Update the epsilon to the first element of the timestep list. In this case it
        # should return the first element of the epsilon list.
        exploration_schedule.decrement_epsilon(50)
        assert exploration_schedule.get_epsilon() == 0.3
        exploration_schedule.decrement_epsilon(100)
        assert exploration_schedule.get_epsilon() == 0.2

        # Update the epsilon value to a timestep between 100 and 200, lets say 150.
        # In this case it should return the 0.5 * (0.2 + 0.6) = 0.4.
        exploration_schedule.decrement_epsilon(150)
        assert exploration_schedule.get_epsilon() == 0.4

        # Update the epsilon value to a timestep after the last element of the timestep
        # list. In this case it should return the final fallback epsilon.
        exploration_schedule.decrement_epsilon(201)
        assert exploration_schedule.get_epsilon() == 0.01

        # Test that the order of epsilon updates does not matter
        exploration_schedule.decrement_epsilon(201)
        exploration_schedule.decrement_epsilon(50)
        assert exploration_schedule.get_epsilon() == 0.3
