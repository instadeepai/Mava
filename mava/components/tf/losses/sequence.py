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

from typing import Type, Union

import tensorflow as tf
import trfl
from acme.tf import losses

from mava.components.tf.networks import DiscreteValuedDistribution
from mava.utils.training_utils import check_rank, combine_dim


def recurrent_n_step_critic_loss(
    q_values: tf.Tensor,
    target_q_values: tf.Tensor,
    rewards: tf.Tensor,
    discounts: tf.Tensor,
    bootstrap_n: int,
    loss_fn: Union[Type[trfl.td_learning], Type[losses.categorical]],
) -> tf.Tensor:

    seq_len = len(rewards[0])
    assert bootstrap_n < seq_len
    if type(q_values) != DiscreteValuedDistribution:
        check_rank([q_values, target_q_values, rewards, discounts], [2, 2, 2, 2])

    # Construct arguments to compute bootstrap target.
    if type(q_values) != DiscreteValuedDistribution:
        q_tm1 = q_values[:, 0:-bootstrap_n]
        q_t = target_q_values[:, bootstrap_n:]

        q_tm1, _ = combine_dim(q_tm1)
        q_t, _ = combine_dim(q_t)
    else:
        q_tm1 = q_values
        q_tm1.cut_dimension(axis=1, end=-bootstrap_n)

        q_t = target_q_values
        q_t.cut_dimension(axis=1, start=bootstrap_n)
    n_step_rewards = rewards[:, :-bootstrap_n]
    n_step_discount = discounts[:, :-bootstrap_n]
    for i in range(1, bootstrap_n + 1):
        n_step_rewards += rewards[:, i : i + seq_len - bootstrap_n]

    n_step_rewards, _ = combine_dim(n_step_rewards)
    n_step_discount, _ = combine_dim(n_step_discount)

    critic_loss = loss_fn(
        q_tm1,
        n_step_rewards,
        n_step_discount,
        q_t,
    )

    if type(q_values) != DiscreteValuedDistribution:
        critic_loss = critic_loss.loss

    return critic_loss
