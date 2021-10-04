import os
from typing import Any, Dict, Iterable, List, Sequence, Tuple, Union

import sonnet as snt
import tensorflow as tf
import trfl


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
