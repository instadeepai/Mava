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

import math

import tensorflow as tf
import trfl

from mava.components.tf.networks import DiscreteValuedDistribution
from mava.utils.training_utils import check_rank, combine_dim


def recurrent_n_step_critic_loss(
    q_values: tf.Tensor,
    target_q_values: tf.Tensor,
    rewards: tf.Tensor,
    bootstrap_n: int,
    discount: float,
    end_of_episode: tf.Tensor,
) -> tf.Tensor:

    if type(q_values) == DiscreteValuedDistribution:
        # Convert to tensor values
        q_values = tf.reshape(q_values._mean(), rewards.shape)
        target_q_values = tf.reshape(target_q_values._mean(), rewards.shape)

    q_values = tf.squeeze(q_values)

    target_q_values = tf.squeeze(target_q_values)
    rewards = tf.squeeze(rewards)
    check_rank([q_values, target_q_values, rewards], [2, 2, 2])

    seq_len = len(rewards[0])

    # d: discount * done
    # bootstrap_n=1 is the normal return of Q_t-1 = R_t-1 + d * Q_t
    # bootstrap_n=seq_len is the Q_t_1 = discounted sum of rewards return
    assert 0 < bootstrap_n <= len(rewards[0])

    # Construct arguments to compute bootstrap target.
    q_tm1 = q_values

    # The last values that rolled over do not matter because a
    # mask is applied to it (hopefully).
    q_t = tf.roll(target_q_values, shift=-bootstrap_n, axis=1)

    # Create the end of episode mask.
    ones_mask = tf.ones(shape=q_t[:, :-bootstrap_n].shape)
    zeros_mask = tf.zeros(shape=q_t[:, -bootstrap_n:].shape)
    eoe_mask = tf.concat([ones_mask, zeros_mask], axis=1)

    q_tm1, _ = combine_dim(q_tm1)
    n_step_rewards = rewards

    # Pad the rewards so that rewards at the end can also be calculated.
    zeros_mask = tf.zeros(shape=rewards[:, : bootstrap_n - 1].shape)
    padded_rewards = tf.concat([rewards, zeros_mask], axis=1)
    for i in range(1, bootstrap_n):
        n_step_rewards += padded_rewards[:, i : i + seq_len] * math.pow(discount, i)
    n_step_rewards, _ = combine_dim(n_step_rewards)

    q_t, _ = combine_dim(q_t)
    eoe_mask, _ = combine_dim(eoe_mask)
    done_eoe_masking, _ = combine_dim(
        tf.roll(end_of_episode, shift=-bootstrap_n, axis=1)
    )

    critic_loss = trfl.td_learning(
        v_tm1=q_tm1,
        r_t=n_step_rewards,
        pcont_t=eoe_mask * done_eoe_masking * math.pow(discount, bootstrap_n),
        v_t=q_t,
    )

    if type(q_t) != DiscreteValuedDistribution:
        critic_loss = critic_loss.loss
    return critic_loss
