from abc import abstractmethod
from typing import Iterator

import dm_env


class SequentialEnvWrapper(dm_env.Environment):
    """
    Abstract class for sequential environment wrappers.
    """

    @abstractmethod
    def agent_iter(self, max_iter: int) -> Iterator:
        """
        Returns an iterator that yields the current agent in the environment.
            max_iter: Maximum amount of iterations (to limit infinite loops/iterations).
        """

    @abstractmethod
    def env_done(self) -> bool:
        """
        Returns a bool indicating if all agents in env are done.
        """


class ParallelEnvWrapper(dm_env.Environment):
    """
    Abstract class for parallel environment wrappers.
    """

    @abstractmethod
    def env_done(self) -> bool:
        """
        Returns a bool indicating if all agents in env are done.
        """
