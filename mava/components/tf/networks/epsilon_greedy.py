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
from typing import Optional

import tensorflow as tf
import tensorflow.compat.v1 as tfv1
import tensorflow_probability as tfp
from acme import types as acme_types


def epsilon_greedy_action_selector(
    action_values: acme_types.NestedArray,
    epsilon: Optional[tf.Tensor] = None,
    legal_actions_mask: acme_types.NestedArray = None,
) -> tfp.distributions.Categorical:
    """Computes an epsilon-greedy distribution over actions.
    This returns a categorical distribution over a discrete action space. It is
    assumed that the trailing dimension of `action_values` is of length A, i.e.
    the number of actions. It is also assumed that actions are 0-indexed.
    This policy does the following:
    - With probability 1 - epsilon, take the action corresponding to the highest
    action value, breaking ties uniformly at random.
    - With probability epsilon, take an action uniformly at random.
    Args:
      action_values: A Tensor of action values with any rank >= 1 and dtype float.
        Shape can be flat ([A]), batched ([B, A]), a batch of sequences
        ([T, B, A]), and so on.
      epsilon: A scalar Tensor (or Python float) with value between 0 and 1.
      legal_actions_mask: An optional one-hot tensor having the shame shape and
        dtypes as `action_values`, defining the legal actions:
        legal_actions_mask[..., a] = 1 if a is legal, 0 otherwise.
        If not provided, all actions will be considered legal and
        `tf.ones_like(action_values)`.
    Returns:
      policy: tfp.distributions.Categorical distribution representing the policy.
    """
    with tfv1.name_scope("epsilon_greedy", values=[action_values, epsilon]):

        # Convert inputs to Tensors if they aren't already.
        action_values = tfv1.convert_to_tensor(action_values)

        if epsilon is not None:
            epsilon = tfv1.convert_to_tensor(epsilon, dtype=action_values.dtype)
        else:
            epsilon = 0

        # convert mask to float
        legal_actions_mask = tfv1.cast(legal_actions_mask, dtype=tf.float32)

        # We compute the action space dynamically.
        num_actions = tfv1.cast(tfv1.shape(action_values)[-1], action_values.dtype)

        # Dithering action distribution.
        if legal_actions_mask is None:
            dither_probs = 1 / num_actions * tfv1.ones_like(action_values)
        else:
            dither_probs = (
                1
                / tfv1.reduce_sum(legal_actions_mask, axis=-1, keepdims=True)
                * legal_actions_mask
            )

        # Greedy action distribution, breaking ties uniformly at random.
        max_value = tfv1.reduce_max(action_values, axis=-1, keepdims=True)
        greedy_probs = tfv1.cast(
            tfv1.equal(action_values, max_value), action_values.dtype
        )
        greedy_probs /= (
            tfv1.reduce_sum(greedy_probs, axis=-1, keepdims=True) * legal_actions_mask
        )

        # Epsilon-greedy action distribution.
        probs = epsilon * dither_probs + (1 - epsilon) * greedy_probs

        # Make the policy object.
        policy = tfp.distributions.Categorical(probs=probs)

    return tf.cast(policy.sample(), "int64")
