from typing import Dict, NamedTuple, Union

import dm_env
import numpy as np
from acme import types
from dm_env import specs


def generate_zeros_from_spec(spec: specs.Array) -> np.ndarray:
    return np.zeros(spec.shape, spec.dtype)


def convert_np_type(dtype: np.dtype, value: Union[int, float]) -> Union[int, float]:
    return np.dtype(dtype).type(value)


def parameterized_restart(
    reward: Union[int, float, types.NestedArray],
    discount: Union[int, float, types.NestedArray],
    observation: Union[Dict, types.NestedArray],
) -> dm_env.TimeStep:
    """Returns a `TimeStep` with `step_type` set to `StepType.FIRST`.
    Differs from dm_env.restart, since reward and discount can be set to
    initial types."""
    return dm_env.TimeStep(dm_env.StepType.FIRST, reward, discount, observation)


class OLT(NamedTuple):
    """Container for (observation, legal_actions, terminal) tuples."""

    observation: types.Nest
    legal_actions: types.Nest
    terminal: types.Nest


class RunningStatistics:
    """Helper class to comute running statistics such as
    the max, min, mean, variance and standard deviation of
    a specific quantity.
    """

    def __init__(self, label: str) -> None:
        self.count = 0
        self.old_mean = 0.0
        self.new_mean = 0.0
        self.old_var = 0.0
        self.new_var = 0.0

        self._max = -9999999.9
        self._min = 9999999.9

        self._label = label

    def push(self, x: float) -> None:
        self.count += 1

        if x > self._max:
            self._max = x

        if x < self._min:
            self._min = x

        if self.count == 1:
            self.old_mean = self.new_mean = x
            self.old_var = 0.0
        else:
            self.new_mean = self.old_mean + (x - self.old_mean) / self.count
            self.new_var = self.old_var + (x - self.old_mean) * (x - self.new_mean)

            self.old_mean = self.new_mean
            self.old_var = self.new_var

    def max(self) -> float:
        return self._max

    def min(self) -> float:
        return self._min

    def mean(self) -> float:
        return self.new_mean if self.count else 0.0

    def variance(self) -> float:
        return self.new_var / (self.count - 1) if self.count > 1 else 0.0

    def standard_deviation(self) -> float:
        return np.sqrt(self.variance())
