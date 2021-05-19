from typing import Type, Union

import tensorflow as tf
import trfl
from acme.tf import losses

from mava.components.tf.networks.mad4pg import DiscreteValuedDistribution
from mava.utils.training_utils import check_rank, combine_dim


def recurrent_n_step_critic_loss(
    q_values: tf.Tensor,
    target_q_values: tf.Tensor,
    rewards: tf.Tensor,
    discounts: tf.Tensor,
    bootstrap_n: int,
    loss_fn: Union[Type[trfl.td_learning], Type[losses.categorical]],
) -> tf.Tensor:

    assert bootstrap_n < len(rewards[0])
    if type(q_values) != DiscreteValuedDistribution:
        check_rank([q_values, target_q_values, rewards, discounts], [2, 2, 2, 2])

    # Construct arguments to compute bootstrap target.
    # TODO (dries): Is the discount calculation correct?

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

    n_step_rewards = rewards[:, :bootstrap_n]
    n_step_discount = discounts[:, :bootstrap_n]
    for i in range(1, bootstrap_n + 1):
        n_step_rewards += rewards[:, i : i + bootstrap_n]

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
