import os
import time
import warnings
from typing import Dict, Optional, Tuple

import tensorflow as tf


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

    valid_options = {
        "trainer_steps",
        "trainer_walltime",
        "evaluator_steps",
        "evaluator_episodes",
        "executor_episodes",
        "executor_steps",
    }

    condition_key, condition_count = None, None
    if condition is not None:
        if len(condition) != 1:
            raise Exception("Please pass only a single termination condition.")
        condition_key, condition_count = list(condition.items())[0]
        if condition_key not in valid_options:
            raise Exception(
                "Please give a valid termination condition. "
                f"Current valid conditions are {valid_options}"
            )
        if not condition_count > 0:
            raise Exception(
                "Termination condition must have a positive value greater than 0."
            )

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
