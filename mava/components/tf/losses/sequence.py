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

from mava.utils.training_utils import check_rank, combine_dim


def recurrent_n_step_critic_loss(
    q_values: tf.Tensor,
    target_q_values: tf.Tensor,
    rewards: tf.Tensor,
    bootstrap_n: int,
    discount: float,
    step_not_padded: tf.Tensor,
) -> tf.Tensor:
    """
    Note:
    bootstrap_n=1 is the normal return of Q_t-1 = R_t-1 + d * Q_t
    bootstrap_n=seq_len is the Q_t_1 = discounted sum of rewards return

    step_not_padded: Indicates whether each step is not padded (0: padded, 1: not padded).

    step_not_padded = tf.roll(dm_end_of_episode[agent], shift=2, axis=1)
    step_not_padded[:,0] = 1.0
    """

    seq_len = len(rewards[0])
    assert 0 < bootstrap_n <= seq_len
    check_rank([q_values, target_q_values, rewards], [3, 3, 2])

    # The last values that rolled over do not matter because a
    # mask is applied to it.
    # tf.print("q_values: ", q_values.shape)
    num_atoms = q_values.shape[-1]

    q_tm1, _ = combine_dim(q_values)
    q_t, _ = combine_dim(tf.roll(target_q_values, shift=-bootstrap_n, axis=1))

    # Pad the rewards so that rewards at the end can also be calculated.
    r_shape = rewards.shape
    zeros_mask = tf.zeros(shape=r_shape[:-1] + (r_shape[-1] - bootstrap_n - 1,))
    padded_rewards = tf.concat([rewards, zeros_mask], axis=1)
    n_step_rewards = rewards
    for i in range(1, bootstrap_n):
        n_step_rewards += padded_rewards[:, i : i + seq_len] * math.pow(discount, i)
    n_step_rewards, _ = combine_dim(n_step_rewards)

    # Create the end of episode mask.
    ones_mask = tf.ones(shape=r_shape[:-1] + (r_shape[-1] - bootstrap_n,))
    zeros_mask = tf.zeros(shape=r_shape[:-1] + (bootstrap_n,))
    fake_experience_mask, _ = combine_dim(tf.concat([ones_mask, zeros_mask], axis=1))

    # Role episode done masking
    step_not_padded_mask, _ = combine_dim(tf.roll(step_not_padded, shift=-bootstrap_n, axis=1))

    flat_mask = fake_experience_mask * step_not_padded_mask * math.pow(discount, bootstrap_n)

    if num_atoms > 1:
        tau = tf.convert_to_tensor(
            [(2 * (i - 1) + 1) / (2 * num_atoms) for i in range(1, num_atoms + 1)]
        )
        # See https://github.com/marload/DistRL-TensorFlow2/blob/master/QR-DQN/QR-DQN.py
        target = (
            tf.expand_dims(n_step_rewards, axis=-1)
            + discount * tf.expand_dims(flat_mask, axis=-1) * q_tm1
        )
        target = tf.stop_gradient(target)
        pred = q_tm1
        pred_tile = tf.tile(tf.expand_dims(pred, axis=2), [1, 1, num_atoms])
        target_tile = tf.tile(tf.expand_dims(target, axis=1), [1, num_atoms, 1])
        huber_loss = tf.compat.v1.losses.huber_loss(target_tile, pred_tile)
        tau = tf.cast(tf.reshape(tau, [1, num_atoms]), dtype="float32")
        inv_tau = 1.0 - tau
        tau = tf.tile(tf.expand_dims(tau, axis=1), [1, num_atoms, 1])
        inv_tau = tf.tile(tf.expand_dims(inv_tau, axis=1), [1, num_atoms, 1])
        error_loss = tf.math.subtract(target_tile, pred_tile)
        loss = tf.where(
            tf.less(error_loss, 0.0), inv_tau * huber_loss, tau * huber_loss
        )
        critic_loss = tf.reduce_sum(tf.reduce_mean(loss, axis=2), axis=1)

        # TODO (dries): This is still incorrect! The masking needs to not be 
        # zero in the case that the episode has already ended and only be zero
        # when the episode has not ended yet.
        ones_mask = tf.ones(shape=r_shape[:-1] + (r_shape[-1] - bootstrap_n,))
        zeros_mask = tf.zeros(shape=r_shape[:-1] + (bootstrap_n,))
        critic_mask, _ = combine_dim(tf.concat([ones_mask, zeros_mask], axis=1))
        critic_loss = critic_loss*critic_mask
        critic_loss = tf.reduce_sum(critic_loss)/tf.reduce_sum(critic_mask)

        # critic_loss = losses.categorical(
        #     q_tm1=q_tm1,
        #     r_t=n_step_rewards,
        #     d_t=flat_mask,
        #     q_t=q_t,
        # )
    else:
        critic_loss = trfl.td_learning(
            v_tm1=q_tm1,
            r_t=n_step_rewards,
            pcont_t=flat_mask,
            v_t=q_t,
        ).loss

    return critic_loss
