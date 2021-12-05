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
import tensorflow as tf


class BaseExplorationScheduler:
    @abc.abstractmethod
    def __init__(
        self,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 1e-4,
    ):
        """Base class for decaying epsilon by schedule.

        Args:
            epsilon_start : initial epsilon value.
            epsilon_min : final epsilon value.
            epsilon_decay : epsilon decay rate, i.e. the decay per executor step.
        """
        self._epsilon_start = epsilon_start
        self._epsilon_min = epsilon_min
        self._epsilon_decay = epsilon_decay
        self._epsilon = epsilon_start

    @abc.abstractmethod
    def decrement_epsilon(self) -> float:
        """Decrement epsilon and return updated epsilon."""

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
        """Decays epsilon linearly to epsilon_min."""
        super(LinearExplorationScheduler, self).__init__(
            epsilon_start,
            epsilon_min,
            epsilon_decay,
        )

    def decrement_epsilon(self) -> float:
        """Decrement/update epsilon.

        Returns:
            current epsilon value.
        """
        self._epsilon = max(self._epsilon_min, self._epsilon - self._epsilon_decay)
        return self._epsilon


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

    def decrement_epsilon(self) -> float:
        """Decrement/update epsilon.

        Returns:
            current epsilon value.
        """
        self._epsilon = max(
            self._epsilon_min, self._epsilon * (1 - self._epsilon_decay)
        )
        return self._epsilon


class BaseExplorationTimestepScheduler:
    @abc.abstractmethod
    def __init__(
        self,
        epsilon_decay_steps: int,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
    ):
        """Base class for decaying epsilon according to number of steps.

        Args:
            epsilon_decay_steps : number of executor steps that epsilon is decayed for.
            epsilon_start : initial epsilon value..
            epsilon_min : final epsilon value.
        """

        self._epsilon_start = epsilon_start
        self._epsilon_min = epsilon_min
        self._epsilon_decay_steps = epsilon_decay_steps
        self._epsilon = epsilon_start

    @abc.abstractmethod
    def decrement_epsilon(self, time_t: int) -> float:
        """Decrement epsilon and return updated epsilon."""

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

    def decrement_epsilon(self, time_t: int) -> float:
        """Decrement/update epsilon.

        Args:
            time_t : executor timestep.
        """
        self._epsilon = max(
            self._epsilon_min, self._epsilon_start - self._delta * time_t
        )
        return self._epsilon


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

    def decrement_epsilon(self, time_t: int) -> float:
        """Decrement/update epsilon.

        Args:
            time_t : executor timestep.
        """
        self._epsilon = min(
            self._epsilon_start,
            max(self._epsilon_min, np.exp(-time_t / self._exp_scaling)),
        )
        return self._epsilon


class ConstantScheduler:
    """Simple scheduler that returns a constant value."""

    def __init__(self, epsilon: float) -> None:
        """Constructors constant scheduler.

        Args:
            epsilon : the constant eps value to be used.
        """
        self._epsilon = tf.constant(epsilon)

    def get_epsilon(self) -> float:
        """Returns constant epsilon.

        Returns:
            constant epsilon.
        """
        return self._epsilon

    def decrement_epsilon(self) -> float:
        """Return constant epsilon.

        Returns:
            constant epsilon.
        """
        return self._epsilon
