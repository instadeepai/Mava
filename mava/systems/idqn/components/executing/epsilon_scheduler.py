from dataclasses import dataclass
from typing import Callable

from mava.components import Component
from mava.core_jax import SystemExecutor
from mava.utils.schedulers import LinearEpsilonScheduler


@dataclass
class EpsilonSchedulerConfig:
    epsilon_scheduler: Callable[[int], float] = LinearEpsilonScheduler(
        initial_epsilon=1.0, final_epsilon=0.1, decay_steps=10_000
    )


class EpsilonScheduler(Component):
    def __init__(
        self, config: EpsilonSchedulerConfig = EpsilonSchedulerConfig()
    ) -> None:
        """Initialises an epsilon scheduler and stores it in the store.

        The scheduler is set through the EpsilonSchedulerConfig and is simply a
            callable: f(steps: int) -> epsilon: float
        """
        super().__init__(config)

    def on_execution_init_end(self, executor: SystemExecutor) -> None:
        """Stores the epsilon scheduler function"""
        executor.store.epsilon_scheduler = self.config.epsilon_scheduler

    @staticmethod
    def name() -> str:
        """Static method that returns component name."""
        return "epsilon_scheduler"
