from typing import Tuple

import chex
import jax.numpy as jnp

from mava.types import ValueNormParams


def _get_running_mean_var(
    value_norm_state: ValueNormParams,
) -> Tuple[chex.Array, chex.Array]:
    debiased_mean = value_norm_state.running_mean / jnp.clip(
        value_norm_state.debiasing_term, a_min=value_norm_state.epsilon, a_max=None
    )
    debiased_mean_sq = value_norm_state.running_mean_sq / jnp.clip(
        value_norm_state.debiasing_term, a_min=value_norm_state.epsilon, a_max=None
    )
    debiased_var = jnp.clip(debiased_mean_sq - debiased_mean**2, a_min=1e-2, a_max=None)
    return debiased_mean, debiased_var


def update_running_mean_var(
    value_norm_state: ValueNormParams, input_batch: chex.Array, value_norm: bool = False
) -> ValueNormParams:

    if value_norm:
        batch_mean = input_batch.mean()
        batch_mean_sq = (input_batch**2).mean()

        new_running_mean = value_norm_state.running_mean * value_norm_state.beta + (
            batch_mean * (1.0 - value_norm_state.beta)
        )
        new_running_mean_sq = (
            value_norm_state.running_mean_sq * value_norm_state.beta
            + (batch_mean_sq * (1.0 - value_norm_state.beta))
        )
        new_debiasing_term = value_norm_state.debiasing_term * value_norm_state.beta + (
            1.0 * (1.0 - value_norm_state.beta)
        )

        value_norm_state = value_norm_state._replace(
            running_mean=new_running_mean,
            running_mean_sq=new_running_mean_sq,
            debiasing_term=new_debiasing_term,
        )

    return value_norm_state


def normalise(
    value_norm_state: ValueNormParams, values: chex.Array, value_norm: bool = False
) -> chex.Array:
    if value_norm:
        debiased_mean, debiased_var = _get_running_mean_var(value_norm_state)
        values = (values - debiased_mean) / jnp.sqrt(debiased_var)
    return values


def denormalise(
    value_norm_state: ValueNormParams, values: chex.Array, value_norm: bool = False
) -> chex.Array:
    if value_norm:
        debiased_mean, debiased_var = _get_running_mean_var(value_norm_state)
        values = values * jnp.sqrt(debiased_var) + debiased_mean
    return values


def reset_state(value_norm_state: ValueNormParams) -> ValueNormParams:
    return value_norm_state._replace(
        running_mean=jnp.zeros_like(value_norm_state.running_mean),
        running_mean_sq=jnp.zeros_like(value_norm_state.running_mean_sq),
        debiasing_term=0.0,
    )