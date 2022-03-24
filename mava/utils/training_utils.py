import os
import time
import warnings
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import trfl

from mava.types import NestedArray


def action_mask_categorical_policies(
    policy: tfp.distributions.Categorical, batched_legal_actions: np.ndarray
) -> tfp.distributions.Categorical:
    """Masks the actions of a categorical policy.

    Args:
        policy : policy to mask.
        batched_legal_actions : batched legal actions.

    Returns:
        masked categorical policy.
    """
    # Mask out actions
    inf_mask = tf.maximum(
        tf.math.log(tf.cast(batched_legal_actions, tf.float32)), tf.float32.min
    )
    masked_logits = policy.logits + inf_mask

    policy = tfp.distributions.Categorical(logits=masked_logits, dtype=policy.dtype)

    return policy


# Adapted from
# https://github.com/tensorflow/agents/blob/f2bebb1e3bc34dc49e34a3d1d6baf30ceee9523c/tf_agents/utils/value_ops.py#L98
def generalized_advantage_estimation(
    values: NestedArray,
    final_value: float,
    discounts: NestedArray,
    rewards: NestedArray,
    td_lambda: float = 1.0,
    time_major: bool = True,
) -> NestedArray:
    """Computes generalized advantage estimation (GAE).

    For theory, see
    "High-Dimensional Continuous Control Using Generalized Advantage Estimation"
    by John Schulman, Philipp Moritz et al.
    See https://arxiv.org/abs/1506.02438 for full paper.
    Define abbreviations:
      (B) batch size representing number of trajectories
      (T) number of steps per trajectory
    Args:
      values: Tensor with shape `[T, B]` representing value estimates.
      final_value: Tensor with shape `[B]` representing value estimate at t=T.
      discounts: Tensor with shape `[T, B]` representing discounts received by
        following the behavior policy.
      rewards: Tensor with shape `[T, B]` representing rewards received by
        following the behavior policy.
      td_lambda: A float32 scalar between [0, 1]. It's used for variance reduction
        in temporal difference.
      time_major: A boolean indicating whether input tensors are time major.
        False means input tensors have shape `[B, T]`.

    Returns:
      A tensor with shape `[T, B]` representing advantages. Shape is `[B, T]` when
      `not time_major`.
    """

    if not time_major:
        with tf.name_scope("to_time_major_tensors"):
            discounts = tf.transpose(discounts)
            rewards = tf.transpose(rewards)
            values = tf.transpose(values)

    with tf.name_scope("gae"):

        next_values = tf.concat([values[1:], tf.expand_dims(final_value, 0)], axis=0)
        delta = rewards + discounts * next_values - values
        weighted_discounts = discounts * td_lambda

        def weighted_cumulative_td_fn(
            accumulated_td: NestedArray, reversed_weights_td_tuple: Tuple
        ) -> NestedArray:
            weighted_discount, td = reversed_weights_td_tuple
            return td + weighted_discount * accumulated_td

        advantages = tf.nest.map_structure(
            tf.stop_gradient,
            tf.scan(
                fn=weighted_cumulative_td_fn,
                elems=(weighted_discounts, delta),
                initializer=tf.zeros_like(final_value),
                reverse=True,
            ),
        )

    if not time_major:
        with tf.name_scope("to_batch_major_tensors"):
            advantages = tf.transpose(advantages)

    return tf.stop_gradient(advantages)


# Adapted from
# https://github.com/tensorflow/agents/blob/f2bebb1e3bc34dc49e34a3d1d6baf30ceee9523c/tf_agents/agents/ppo/ppo_agent.py#L100
def _normalize_advantages(
    advantages: NestedArray, axes: Tuple = (0, 1), variance_epsilon: float = 1e-8
) -> NestedArray:
    """Normalize advantage estimate.

    Args:
        advantages : advantages.
        axes : axes for normalization.
        variance_epsilon : variance used to prevent dividing by zero.

    Returns:
        normalized advantages.
    """
    adv_mean, adv_var = tf.nn.moments(advantages, axes=axes, keepdims=True)
    normalized_advantages = tf.nn.batch_normalization(
        advantages,
        adv_mean,
        adv_var,
        offset=None,
        scale=None,
        variance_epsilon=variance_epsilon,
    )
    return normalized_advantages


def decay_lr_actor_critic(
    learning_rate_scheduler_fn: Optional[Dict[str, Callable[[int], None]]],
    policy_optimizers: Dict,
    critic_optimizers: Dict,
    trainer_step: int,
) -> None:
    """Function that decays lr rate in actor critic training.

    Args:
        learning_rate_scheduler_fn : dict of functions (for policy and critic networks),
            that return a learning rate at training time t.
        policy_optimizers : policy optims.
        critic_optimizers : critic optims.
        trainer_step : training time t.
    """
    if learning_rate_scheduler_fn:
        if learning_rate_scheduler_fn["policy"]:
            decay_lr(
                learning_rate_scheduler_fn["policy"], policy_optimizers, trainer_step
            )

        if learning_rate_scheduler_fn["critic"]:
            decay_lr(
                learning_rate_scheduler_fn["critic"], critic_optimizers, trainer_step
            )


def decay_lr(
    lr_schedule: Optional[Callable], optimizers: Dict, trainer_step: int
) -> None:
    """Funtion that decays lr of optim.

    Args:
        lr_schedule : lr schedule function/callable.
        optimizers : optim to decay.
        trainer_step : training time t.
    """
    if lr_schedule and callable(lr_schedule):
        lr = lr_schedule(trainer_step)
        for optimizer in optimizers.values():
            optimizer.learning_rate = lr


def non_blocking_sleep(time_in_seconds: int) -> None:
    """Function to sleep for time_in_seconds, without hanging lp program.

    Args:
        time_in_seconds : number of seconds to sleep for.
    """
    for _ in range(time_in_seconds):
        # Do not sleep for a long period of time to avoid LaunchPad program
        # termination hangs (time.sleep is not interruptible).
        time.sleep(1)


def check_count_condition(condition: Optional[dict]) -> Tuple:
    """Checks if condition is valid.

    These conditions are used for termination or to run evaluators in intervals.

    Args:
        condition : a dict with a key referring to the name of a condition and the
        value referring to count of the condition that needs to be reached.
        e.g. {"executor_episodes": 100}

    Returns:
        the condition key and count.
    """

    valid_options = [
        "trainer_steps",
        "trainer_walltime",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    ]

    condition_key, condition_count = None, None
    if condition is not None:
        assert len(condition) == 1
        condition_key, condition_count = list(condition.items())[0]
        assert condition_key in valid_options
        assert condition_count > 0

    return condition_key, condition_count


# Checkpoint the networks.
def checkpoint_networks(system_checkpointer: Dict) -> None:
    """Checkpoint networks.

    Args:
        system_checkpointer : checkpointer used by the system.
    """
    try:
        if system_checkpointer and len(system_checkpointer.keys()) > 0:
            for network_key in system_checkpointer.keys():
                checkpointer = system_checkpointer[network_key]
                checkpointer.save()
    except Exception as ex:
        warnings.warn(f"Failed to checkpoint. Error: {ex}")


def set_growing_gpu_memory() -> None:
    """Solve gpu mem issues."""
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)


# Map critic and polic losses to dict, grouped by agent.
def map_losses_per_agent_ac(
    critic_losses: Dict, policy_losses: Dict, total_losses: Optional[Dict] = None
) -> Dict:
    """Map seperate losses dict to loss per agent.

    Args:
        critic_losses : critic loss per agent.
        policy_losses : policy loss per agent.
        total_losses : optional total (critic + policy loss) per agent.

    Returns:
        dict with losses grouped per agent.
    """
    assert len(policy_losses) > 0 and (
        len(critic_losses) == len(policy_losses)
    ), "Invalid System Checkpointer."
    logged_losses: Dict[str, Dict[str, Any]] = {}
    for agent in policy_losses.keys():
        logged_losses[agent] = {
            "critic_loss": critic_losses[agent],
            "policy_loss": policy_losses[agent],
        }
        if total_losses is not None:
            logged_losses[agent]["total_loss"] = total_losses[agent]

    return logged_losses


def map_losses_per_agent_value(value_losses: Dict) -> Dict:
    """Map value losses to dict, grouped by agent.

    Args:
        value_losses : value losses.

    Returns:
        losses grouped per agent.
    """
    assert len(value_losses) > 0, "Invalid System Checkpointer."
    logged_losses: Dict[str, Dict[str, Any]] = {}
    for agent in value_losses.keys():
        logged_losses[agent] = {
            "value_loss": value_losses[agent],
        }

    return logged_losses


def combine_dim(inputs: Union[tf.Tensor, List, Tuple]) -> tf.Tensor:
    """Merge dims.

    Args:
        inputs : input tensor/list.

    Returns:
        tensor with dims merged.
    """
    if isinstance(inputs, tf.Tensor):
        dims = inputs.shape[:2]
        return snt.merge_leading_dims(inputs, num_dims=2), dims
    else:
        dims = None
        return_list = []
        for tensor in inputs:
            comb_one, dims = combine_dim(tensor)
            return_list.append(comb_one)
        return return_list, dims


def extract_dim(inputs: Union[tf.Tensor, List, Tuple], dims: tf.Tensor) -> tf.Tensor:
    """Reshape or extract dim of tensor.

    Args:
        inputs : input tensor/list.
        dims : dim.

    Returns:
        reshaped or extracted dim.
    """
    if isinstance(inputs, tf.Tensor):
        return tf.reshape(inputs, [dims[0], dims[1], -1])
    else:
        return [extract_dim(tensor, dims) for tensor in inputs]


def check_rank(tensors: Iterable[tf.Tensor], ranks: Sequence[int]) -> None:
    """Check rank.

    Require correct tensor ranks---as long as we have shape information,
    available to check. If there isn't any, we print a warning.

    Args:
        tensors : _description_
        ranks : _description_

    Raises:
        ValueError: _description_
    """
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


def safe_del(object_class: Any, attrname: str) -> None:
    """Safely delete object from class.

    Args:
        object_class : object closs.
        attrname : name of attr to delete.
    """
    try:
        if hasattr(object_class, attrname):
            delattr(object_class, attrname)
    except AttributeError:
        pass
