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
