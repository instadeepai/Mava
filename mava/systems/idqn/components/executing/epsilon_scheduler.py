from dataclasses import dataclass
from typing import Callable

from mava.components import Component
from mava.core_jax import SystemExecutor
from mava.utils.schedulers import LinearEpsilonScheduler


@dataclass
class EpsilonSchedulerConfig:
    epsilon_scheduler: Callable[[int], float] = LinearEpsilonScheduler(
        initial_epsilon=1.0, final_epsilon=0.05, decay_steps=50_000
    )


class EpsilonScheduler(Component):
    def __init__(
        self, config: EpsilonSchedulerConfig = EpsilonSchedulerConfig()
    ) -> None:
        super().__init__(config)

    def on_execution_init_end(self, executor: SystemExecutor):
        executor.store.epsilon_scheduler = self.config.epsilon_scheduler

    @staticmethod
    def name():
        return "epsilon_scheduler"
