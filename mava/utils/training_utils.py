import os
from typing import Any, Callable, Dict, Iterable, Optional, Sequence

import sonnet as snt
import tensorflow as tf
import trfl


def decay_lr_actor_critic(
    learning_rate_schedule: Optional[Dict[str, Callable[[int], None]]],
    policy_optimizers: Dict,
    critic_optimizers: Dict,
    trainer_step: int,
) -> None:
    """Function that decays lr rate in actor critic training.

    Args:
        learning_rate_schedule : dict of functions (for policy and critic networks),
            that return a learning rate at training time t.
        policy_optimizers : policy optims.
        critic_optimizers : critic optims.
        trainer_step : training time t.
    """
    if learning_rate_schedule:
        if learning_rate_schedule["policy"]:
            decay_lr(learning_rate_schedule["policy"], policy_optimizers, trainer_step)

        if learning_rate_schedule["critic"]:
            decay_lr(learning_rate_schedule["critic"], critic_optimizers, trainer_step)


def decay_lr_actor(
    learning_rate_schedule: Optional[Callable[[int], None]],
    policy_optimizers: Dict,
    trainer_step: int,
) -> None:
    """Function that decays lr rate in actor training.

    Args:
        learning_rate_schedule : function that returns a learning rate at training
            time t.
        policy_optimizers : policy optims.
        trainer_step : training time t.
    """
    if learning_rate_schedule:
        decay_lr(learning_rate_schedule, policy_optimizers, trainer_step)


def decay_lr(
    lr_schedule: Callable[[int], None], optimizers: Dict, trainer_step: int
) -> None:
    """Funtion that decays lr of optim.

    Args:
        lr_schedule : lr schedule function.
        optimizer : optim to decay.
        trainer_step : training time t.
    """
    if lr_schedule and callable(lr_schedule):
        lr = lr_schedule(trainer_step)
        for optimizer in optimizers.values():
            optimizer.learning_rate = lr


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


# Safely delete object from class.
def safe_del(object_class: Any, attrname: str) -> None:
    try:
        if hasattr(object_class, attrname):
            delattr(object_class, attrname)
    except AttributeError:
        pass
