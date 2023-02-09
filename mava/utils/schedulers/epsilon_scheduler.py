from abc import ABC, abstractmethod


class EpsilonScheduler(ABC):
    @abstractmethod
    def __call__(self, step: int) -> float:
        """Returns the value of epsilon at the current step"""
