import os
import time
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import sonnet as snt
import tensorflow as tf
import trfl


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
        optimizer : optim to decay.
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
    """Checks if condition is valid. These conditions are used for termination
    or to run evaluators in intervals.

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
    if system_checkpointer and len(system_checkpointer.keys()) > 0:
        for network_key in system_checkpointer.keys():
            checkpointer = system_checkpointer[network_key]
            checkpointer.save()


def set_growing_gpu_memory() -> None:
    # Solve gpu mem issues.
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    physical_devices = tf.config.list_physical_devices("GPU")
    if physical_devices:
        for device in physical_devices:
            tf.config.experimental.set_memory_growth(device, True)


# Map critic and polic losses to dict, grouped by agent.
def map_losses_per_agent_ac(critic_losses: Dict, policy_losses: Dict) -> Dict:
    assert len(policy_losses) > 0 and (
        len(critic_losses) == len(policy_losses)
    ), "Invalid System Checkpointer."
    logged_losses: Dict[str, Dict[str, Any]] = {}
    for agent in policy_losses.keys():
        logged_losses[agent] = {
            "critic_loss": critic_losses[agent],
            "policy_loss": policy_losses[agent],
        }

    return logged_losses


# Map value losses to dict, grouped by agent.
def map_losses_per_agent_value(value_losses: Dict) -> Dict:
    assert len(value_losses) > 0, "Invalid System Checkpointer."
    logged_losses: Dict[str, Dict[str, Any]] = {}
    for agent in value_losses.keys():
        logged_losses[agent] = {
            "value_loss": value_losses[agent],
        }

    return logged_losses


def combine_dim(inputs: Union[tf.Tensor, List, Tuple]) -> tf.Tensor:
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
    if isinstance(inputs, tf.Tensor):
        return tf.reshape(inputs, [dims[0], dims[1], -1])
    else:
        return [extract_dim(tensor, dims) for tensor in inputs]


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


# Safely delete object from class.
def safe_del(object_class: Any, attrname: str) -> None:
    try:
        if hasattr(object_class, attrname):
            delattr(object_class, attrname)
    except AttributeError:
        pass
