# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc

import numpy as np


def faster_max(x: float, y: float) -> float:
    """Max function faster than Max(), since Max() does type checks.


    Args:
        x : var 1.
        y : var 2.

    Returns:
        laegest number between x and y.
    """
    if x > y:
        return x
    else:
        return y


def faster_min(x: float, y: float) -> float:
    """Min function faster than Min(), since Min() does type checks.


    Args:
        x : var 1.
        y : var 2.

    Returns:
        smallest number between x and y.
    """
    if x < y:
        return x
    else:
        return y


class BaseExplorationScheduler:
    @abc.abstractmethod
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 1e-4,
    ):
        """Base class for decaying epsilon by schedule."""
        self._epsilon_start = epsilon_start
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._epsilon = epsilon_start

        # _reached_min is used to improve efficiency.
        self._reached_min = False

    def decrement_epsilon(self) -> float:
        """Decrement epsilon or return current epsilon.

        Returns:
            current epsilon value.
        """
        # If we have reached the minimum value, don't do a min/max
        # operation, just return epsilon.
        if self._reached_min:
            return self._epsilon

        self._decrement_epsilon()

        self._reached_min = self._epsilon == self._epsilon_min

        return self._epsilon

    @abc.abstractmethod
    def _decrement_epsilon(self) -> None:
        """Internal method to decrement/update epsilon."""

    def get_epsilon(self) -> float:
        """Get epsilon value.

        Returns:
            current epsilon.
        """
        return self._epsilon

    def reset_epsilon(self) -> None:
        """Reset epsilon value."""
        self._epsilon = self._epsilon_start


class LinearExplorationScheduler(BaseExplorationScheduler):
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 1e-4,
    ):
        """
        Decays epsilon linearly to epsilon_min.
        """
        super(LinearExplorationScheduler, self).__init__(
            epsilon_start,
            epsilon_min,
            epsilon_decay,
        )

    def _decrement_epsilon(self) -> None:
        self._epsilon = faster_max(
            self._epsilon_min, self._epsilon - self._epsilon_decay
        )


class ExponentialExplorationScheduler(BaseExplorationScheduler):
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 1e-4,
    ):
        """Decays epsilon exponentially to epsilon_min."""
        super(ExponentialExplorationScheduler, self).__init__(
            epsilon_start,
            epsilon_min,
            epsilon_decay,
        )

    def _decrement_epsilon(self) -> None:
        """Decrement/update epsilon.

        Returns:
            current epsilon value.
        """
        self._epsilon = faster_max(
            self._epsilon_min, self._epsilon * (1 - self._epsilon_decay)
        )


class BaseExplorationTimestepScheduler:
    @abc.abstractmethod
    def __init__(
        self,
        epsilon_decay_steps: int,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
    ):
        """Base class for decaying epsilon according to number of steps."""
        self._epsilon_start = epsilon_start
        self._epsilon_min = epsilon_min
        self._epsilon_decay_steps = epsilon_decay_steps
        self._epsilon = epsilon_start

        # _reached_min is used to improve efficiency.
        self._reached_min = False

    def decrement_epsilon(self, time_t: int) -> float:
        """Decrement epsilon or return current epsilon.

        Returns:
            current epsilon value.
        """
        # If we have reached the minimum value, don't do a min/max
        # operation, just return epsilon.
        if self._reached_min:
            return self._epsilon

        self._decrement_epsilon(time_t)

        self._reached_min = self._epsilon == self._epsilon_min

        return self._epsilon

    @abc.abstractmethod
    def _decrement_epsilon(self, time_t: int) -> None:
        """Internal method to decrement/update epsilon."""

    def get_epsilon(self) -> float:
        """Get epsilon value.

        Returns:
            current epsilon.
        """
        return self._epsilon

    def reset_epsilon(self) -> None:
        """Reset epsilon value."""
        self._epsilon = self._epsilon_start


class LinearExplorationTimestepScheduler(BaseExplorationTimestepScheduler):
    def __init__(
        self,
        epsilon_decay_steps: int,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
    ):
        """Decays epsilon linearly to epsilon_min, in epsilon_decay_steps."""
        super(LinearExplorationTimestepScheduler, self).__init__(
            epsilon_decay_steps,
            epsilon_start,
            epsilon_min,
        )

        self._delta = (
            self._epsilon_start - self._epsilon_min
        ) / self._epsilon_decay_steps

    def _decrement_epsilon(self, time_t: int) -> None:
        """Decrement/update epsilon.

        Returns:
            current epsilon value.
        """
        self._epsilon = faster_max(
            self._epsilon_min, self._epsilon_start - self._delta * time_t
        )


# Adapted from
# https://github.com/oxwhirl/pymarl/blob/master/src/components/epsilon_schedules.py
class ExponentialExplorationTimestepScheduler(BaseExplorationTimestepScheduler):
    def __init__(
        self,
        epsilon_decay_steps: int,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
    ):
        """Decays epsilon exponentially to epsilon_min, in epsilon_decay_steps."""
        super(ExponentialExplorationTimestepScheduler, self).__init__(
            epsilon_decay_steps,
            epsilon_start,
            epsilon_min,
        )

        self._exp_scaling = (
            (-1) * self._epsilon_decay_steps / np.log(self._epsilon_min)
            if self._epsilon_min > 0
            else 1
        )

    def _decrement_epsilon(self, time_t: int) -> None:
        """Decrement/update epsilon.

        Returns:
            current epsilon value.
        """
        self._epsilon = faster_min(
            self._epsilon_start,
            faster_max(self._epsilon_min, np.exp(-time_t / self._exp_scaling)),
        )


class ConstantScheduler:
    """Simple scheduler that returns a constant value."""

    def __init__(self, epsilon_start: float) -> None:
        self._epsilon_start = epsilon_start

    def get_epsilon(self) -> float:
        """Returns constant epsilon.

        Returns:
            constant epsilon.
        """
        return self._epsilon_start

    def decrement_epsilon(self) -> float:
        """Return constant epsilon.

        Returns:
            constant epsilon.
        """
        return self._epsilon_start
