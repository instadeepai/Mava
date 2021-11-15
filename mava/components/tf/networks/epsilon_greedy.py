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

# based on the following:
# https://github.com/deepmind/acme/blob/master/acme/tf/networks/masked_epsilon_greedy.py

"""Adaptation of trfl epsilon_greedy with legal action masking."""
import typing
from typing import Optional, Union

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp

from mava.components.tf.modules.exploration.exploration_scheduling import (
    BaseExplorationScheduler,
    BaseExplorationTimestepScheduler,
)


# Adapted from
# https://github.com/deepmind/acme/blob/master/acme/tf/networks/legal_actions.py
class EpsilonGreedy(snt.Module):
    """Computes an epsilon-greedy distribution over actions.

    This policy does the following:
    - With probability 1 - epsilon, take the action corresponding to the highest
    action value, breaking ties uniformly at random.
    - With probability epsilon, take an action uniformly at random.
    """

    def __init__(
        self,
        exploration_scheduler: Union[
            BaseExplorationScheduler, BaseExplorationTimestepScheduler
        ],
        name: str = "EpsilonGreedy",
        seed: Optional[int] = None,
    ):
        """Initialize the action selector.

        Args:
            exploration_scheduler : scheduler for epsilon.
            name : sonnet module name.
            seed: seed for reproducible sampling.
        """

        super().__init__(name=name)
        self._exploration_scheduler = exploration_scheduler
        self._epsilon = tf.Variable(
            self._exploration_scheduler.get_epsilon(), trainable=False
        )
        self._seed = seed

    def __call__(
        self,
        action_values: tf.Tensor,
        legal_actions_mask: Optional[tf.Tensor],
    ) -> tfp.distributions.Categorical:
        """Forward pass of action selector.

        Args:
            action_values: A Tensor of action values with any rank >= 1 and dtype float.
                Shape can be flat ([A]), batched ([B, A]), a batch of sequences
                    ([T, B, A]), and so on.
            legal_actions_mask: An optional one-hot tensor having the shame shape and
                dtypes as `action_values`, defining the legal actions:
                legal_actions_mask[..., a] = 1 if a is legal, 0 otherwise.
                If not provided, all actions will be considered legal and
                `tf.ones_like(action_values)`.

        Returns:
            a sampled action from tf distribution representing the policy.
        """
        if legal_actions_mask is None:
            # We compute the action space dynamically.
            num_actions = tf.cast(tf.shape(action_values)[-1], action_values.dtype)

            # Dithering action distribution.
            dither_probs = 1 / num_actions * tf.ones_like(action_values)
        else:
            legal_actions_mask = tf.cast(legal_actions_mask, dtype=tf.float32)

            # Dithering action distribution.
            dither_probs = (
                1
                / tf.reduce_sum(legal_actions_mask, axis=-1, keepdims=True)
                * legal_actions_mask
            )

        masked_action_values = tf.where(
            tf.equal(legal_actions_mask, 1),
            action_values,
            tf.fill(tf.shape(action_values), -np.inf),
        )

        # Greedy action distribution, breaking ties uniformly at random.
        # Max value considers only valid/masked action values
        max_value = tf.reduce_max(masked_action_values, axis=-1, keepdims=True)
        greedy_probs = tf.cast(
            tf.equal(masked_action_values, max_value),
            action_values.dtype,
        )
        greedy_probs /= tf.reduce_sum(greedy_probs, axis=-1, keepdims=True)

        # Epsilon-greedy action distribution.
        probs = self._epsilon * dither_probs + (1 - self._epsilon) * greedy_probs

        # Make the policy object.
        policy = tfp.distributions.Categorical(probs=probs)

        if self._seed:
            action = policy.sample(seed=self._seed)
        else:
            action = policy.sample()
        # Return sampled action.
        return tf.cast(action, "int64")

    def get_epsilon(self) -> float:
        """Return current epsilon.

        Returns:
            current epsilon.
        """
        return self._epsilon

    # mypy doesn't handle vars with multiple possible types well.
    @typing.no_type_check
    def decrement_epsilon(self) -> None:
        """Decrement epsilon acording to schedule."""
        self._epsilon.assign(self._exploration_scheduler.decrement_epsilon())

    @typing.no_type_check
    def decrement_epsilon_time_t(self, time_t: int) -> None:
        """Decrement epsilon acording to time t."""
        self._epsilon.assign(self._exploration_scheduler.decrement_epsilon(time_t))
