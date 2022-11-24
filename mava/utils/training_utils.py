import os
import time
import warnings
from typing import Dict, Optional, Tuple

import chex
import jax
import jax.numpy as jnp
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


def clipped_surrogate_pg_loss(
    prob_ratios_t: jnp.array,
    adv_t: jnp.array,
    epsilon: float,
    loss_masks: jnp.array,
    use_stop_gradient: bool = True,
) -> jnp.array:
    """
    Modified from: https://github.com/deepmind/rlax/blob/master/rlax/_src/policy_gradients.py
    Computes the clipped surrogate policy gradient loss.
    L_clipₜ(θ) = - min(rₜ(θ)Âₜ, clip(rₜ(θ), 1-ε, 1+ε)Âₜ)
    Where rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ) and Âₜ are the advantages.
    See Proximal Policy Optimization Algorithms, Schulman et al.:
    https://arxiv.org/abs/1707.06347
    Args:
    prob_ratios_t: Ratio of action probabilities for actions a_t:
        rₜ(θ) = π_θ(aₜ| sₜ) / π_θ_old(aₜ| sₜ)
    adv_t: the observed or estimated advantages from executing actions a_t.
    epsilon: Scalar value corresponding to how much to clip the objecctive.
    use_stop_gradient: bool indicating whether or not to apply stop gradient to
        advantages.
    Returns:
    Loss whose gradient corresponds to a clipped surrogate policy gradient
        update.
    """
    chex.assert_rank([prob_ratios_t, adv_t], [1, 1])
    chex.assert_type([prob_ratios_t, adv_t], [float, float])

    adv_t = jax.lax.select(use_stop_gradient, jax.lax.stop_gradient(adv_t), adv_t)
    clipped_ratios_t = jnp.clip(prob_ratios_t, 1.0 - epsilon, 1.0 + epsilon)
    clipped_objective = jnp.fmin(prob_ratios_t * adv_t, clipped_ratios_t * adv_t)
    return -(jnp.sum(clipped_objective * loss_masks) / jnp.sum(loss_masks))


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
