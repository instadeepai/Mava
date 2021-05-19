from typing import Any, Dict, Iterable, Sequence, Type, Union

import sonnet as snt
import tensorflow as tf
import trfl
from acme.tf import losses

from mava.components.tf.networks.mad4pg import DiscreteValuedDistribution


# Checkpoint the networks.
def checkpoint_networks(system_checkpointer: Dict) -> None:
    if system_checkpointer and len(system_checkpointer.keys()) > 0:
        for network_key in system_checkpointer.keys():
            checkpointer = system_checkpointer[network_key]
            checkpointer.save()


# Map critic and polic losses to dict, grouped by agent.
def map_losses_per_agent_ac(critic_losses: Dict, policy_losses: Dict) -> Dict:
    assert len(policy_losses) > 0 and (
        len(critic_losses) == len(policy_losses)
    ), "Invalid System Checkpointer."
    logged_losses: Dict[str, Dict[str, Any]] = {}
    agents = policy_losses.keys()
    for agent in agents:
        logged_losses[agent] = {
            "critic_loss": critic_losses[agent],
            "policy_loss": policy_losses[agent],
        }

    return logged_losses


def combine_dim(tensor: tf.Tensor) -> tf.Tensor:
    dims = tensor.shape[:2]
    return snt.merge_leading_dims(tensor, num_dims=2), dims


def extract_dim(tensor: tf.Tensor, dims: tf.Tensor) -> tf.Tensor:
    return tf.reshape(tensor, [dims[0], dims[1], -1])


# Require correct tensor ranks---as long as we have shape information
# available to check. If there isn't any, we print a warning.
def check_rank(tensors: Iterable[tf.Tensor], ranks: Sequence[int]) -> None:
    for i, (tensor, rank) in enumerate(zip(tensors, ranks)):
        if tensor.get_shape():
            trfl.assert_rank_and_shape_compatibility([tensor], rank)
        else:
            raise ValueError(
                f'Tensor "{tensor.name}", which was offered as '
                f"transformed_n_step_loss parameter {i+1}, has "
                f"no rank at construction time, so cannot verify"
                f"that it has the necessary rank of {rank}"
            )


def maddp4g_critic(
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
        # n_step_discount *= discounts[:, i:i+bootstrap_n]

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


# Safely delete object from class.
def safe_del(object_class: Any, attrname: str) -> None:
    try:
        if hasattr(object_class, attrname):
            delattr(object_class, attrname)
    except AttributeError:
        pass
