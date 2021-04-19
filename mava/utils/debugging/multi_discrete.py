# Adapted from https://github.com/openai/multiagent-particle-envs.
# TODO (dries): Try using this class directly from PettingZoo and delete this file.

# An old version of OpenAI Gym's multi_discrete.py.
# (Was getting affected by Gym updates)
# (https://github.com/openai/gym/blob/1fb81d4e3fb780ccf77fec731287ba07da35eb84
# /gym/spaces/multi_discrete.py)

from typing import List, Tuple

import gym
import numpy as np


class MultiDiscrete(gym.Space):
    """
    - The multi-discrete action space consists of a series
    of discrete action spaces with different parameters
    - It can be adapted to both a Discrete action space or
    a continuous (Box) action space
    - It is useful to represent game controllers or keyboards
    where each key can be represented as a discrete action space
    - It is parametrized by passing an array of arrays containing
    [min, max] for each discrete action space
       where the discrete action space can take any integers
       from `min` to `max` (both inclusive)
    Note: A value of 0 always need to represent the NOOP action.
    e.g. Nintendo Game Controller
    - Can be conceptualized as 3 discrete action spaces:
        1) Arrow Keys: Discrete 5  - NOOP[0], UP[1], RIGHT[2],
        DOWN[3], LEFT[4]  - params: min: 0, max: 4
        2) Button A:   Discrete 2  - NOOP[0], Pressed[1] -
        params: min: 0, max: 1
        3) Button B:   Discrete 2  - NOOP[0], Pressed[1] -
        params: min: 0, max: 1
    - Can be initialized as
        MultiDiscrete([ [0,4], [0,1], [0,1] ])
    """

    def __init__(self, array_of_param_array: np.array) -> None:
        self.low = np.array([x[0] for x in array_of_param_array])
        self.high = np.array([x[1] for x in array_of_param_array])
        self.num_discrete_space = self.low.shape[0]

    def sample(self) -> List[int]:
        """Returns a array with one sample from each discrete action space.
        For each row:  round(random .* (max - min) + min, 0)"""
        random_array = np.random.RandomState().rand(self.num_discrete_space)
        return [
            int(x)
            for x in np.floor(
                np.multiply((self.high - self.low + 1.0), random_array) + self.low
            )
        ]

    def contains(self, x: List) -> Tuple[bool]:
        return (
            len(x) == self.num_discrete_space
            and (np.array(x) >= self.low).all()
            and (np.array(x) <= self.high).all()
        )

    @property
    def shape(self) -> Tuple:
        return self.num_discrete_space

    def __repr__(self) -> str:
        return "MultiDiscrete" + str(self.num_discrete_space)
