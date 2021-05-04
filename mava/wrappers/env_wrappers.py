from abc import abstractmethod
from typing import Any, Iterator, List

import dm_env


class ParallelEnvWrapper(dm_env.Environment):
    """
    Abstract class for parallel environment wrappers.
    """

    @abstractmethod
    def env_done(self) -> bool:
        """
        Returns a bool indicating if all agents in env are done.
        """

    @property
    @abstractmethod
    def agents(self) -> List:
        """
        Returns the active agents in the env.
        """

    @property
    @abstractmethod
    def possible_agents(self) -> List:
        """
        Returns all the possible agents in the env.
        """


class SequentialEnvWrapper(ParallelEnvWrapper):
    """
    Abstract class for sequential environment wrappers.
    """

    @abstractmethod
    def agent_iter(self, max_iter: int) -> Iterator:
        """
        Returns an iterator that yields the current agent in the env.
            max_iter: Maximum number of iterations (to limit infinite loops/iterations).
        """

    @property
    @abstractmethod
    def current_agent(self) -> Any:
        """
        Returns the current selected agent.
        """
