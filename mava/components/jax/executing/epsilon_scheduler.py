from dataclasses import dataclass
from typing import Callable

from mava.components.jax import Component
from mava.core_jax import SystemExecutor
from mava.utils.schedules.linear_epsilon_scheduler import LinearEpsilonScheduler


# TODO: This class is not tested separately.
@dataclass
class EpsilonSchedulerConfig:
    """Configuration for the EpsilonScheduler.

    Notes:
        Requires an instance of EpsilonScheduler. Here a linear epsilon scheduler
        with default parameters of a linear scheduler in RLlib is used.

        This choice is overwritten by the user if necessary via passing another
        scheduler to the parameter epsilon_scheduler in system.build. For an
        example refer to system_test.py for systems.jax.madqn.system_test .
    """

    epsilon_scheduler: LinearEpsilonScheduler = LinearEpsilonScheduler(
        initial_epsilon=1.0, final_epsilon=0.05, decay_steps=10000
    )


class EpsilonScheduler(Component):
    def __init__(self, config: EpsilonSchedulerConfig = EpsilonSchedulerConfig()):
        """Epsilon Scheduler Component.

        Args:
            config : should have an epsilon scheduler which has the following methods:
                update(step): to update the epsilon value
                epsilon: to get the current epsilon value
        """
        self.config = config

    def on_execution_init_end(self, executor: SystemExecutor) -> None:
        """Stores the epsilon scheduler in the store."""
        executor.store.epsilon_scheduler = self.config.epsilon_scheduler

    def on_execution_update(self, executor: SystemExecutor) -> None:
        """Updates the epsilon value for the epsilon scheduler."""
        total_steps = executor.store.executor_counts[
            "executor_steps"
        ]  # executor.store.steps_count
        executor.store.epsilon_scheduler.update(step=total_steps)

    @staticmethod
    def config_class() -> Callable:
        """Returns the configuration class."""
        return EpsilonSchedulerConfig

    @staticmethod
    def name() -> str:
        """Returns the name of the component."""
        return "executor_scheduler"
