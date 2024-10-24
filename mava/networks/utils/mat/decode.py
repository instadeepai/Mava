# Copyright 2022 InstaDeep Ltd. All rights reserved.
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

from typing import Tuple, Union

import chex
import jax
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from flax import linen as nn

from mava.networks.distributions import IdentityTransformation, TanhTransformedDistribution

# General shapes legend:
# B: batch size
# N: number of agents
# O: observation dimension
# A: action dimension
# E: model embedding dimension


def discrete_parallel_act(
    decoder: nn.Module,
    obs_rep: chex.Array,  # (B, N, E)
    action: chex.Array,  # (B, N)
    action_dim: int,  # (, )
    legal_actions: chex.Array,  # (B, N, A)
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array]:
    B, N, _ = obs_rep.shape
    one_hot_action = jax.nn.one_hot(action, action_dim)  # (B, A)
    shifted_action = jnp.zeros((B, N, action_dim + 1))  # (B, N, A +1)
    shifted_action = shifted_action.at[:, 0, 0].set(1)
    shifted_action = shifted_action.at[:, 1:, 1:].set(one_hot_action[:, :-1, :])
    logit = decoder(shifted_action, obs_rep)  # (B, N, A)

    masked_logits = jnp.where(
        legal_actions,
        logit,
        jnp.finfo(jnp.float32).min,
    )

    distribution = IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))
    action_log_prob = distribution.log_prob(action)
    entropy = distribution.entropy(seed=key)

    return action_log_prob, entropy  # (B, N), (B, N)


def continuous_parallel_act(
    decoder: nn.Module,
    obs_rep: chex.Array,  # (B, N, E)
    action: chex.Array,  # (B, N, A)
    action_dim: int,  # (, )
    legal_actions: chex.Array,  # (B, N, A)
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array]:
    # We don't need legal_actions for continuous actions but keep it to keep the APIs consistent.
    del legal_actions
    B, N, _ = obs_rep.shape
    shifted_action = jnp.zeros((B, N, action_dim))

    shifted_action = shifted_action.at[:, 1:, :].set(action[:, :-1, :])

    act_mean = decoder(shifted_action, obs_rep)  # (B, N, A)
    action_std = jax.nn.softplus(decoder.log_std)

    distribution = tfd.Normal(loc=act_mean, scale=action_std)
    distribution = tfd.Independent(
        TanhTransformedDistribution(distribution),
        reinterpreted_batch_ndims=1,
    )
    action_log_prob = distribution.log_prob(action)
    entropy = distribution.entropy(seed=key)

    return action_log_prob, entropy  # (B, N), (B, N)


def discrete_autoregressive_act(
    decoder: nn.Module,
    obs_rep: chex.Array,  # (B, N, E)
    action_dim: int,  # (, )
    legal_actions: chex.Array,  # (B, N, A)
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array]:
    B, N, _ = obs_rep.shape
    shifted_action = jnp.zeros((B, N, action_dim + 1))
    shifted_action = shifted_action.at[:, 0, 0].set(1)
    output_action = jnp.zeros((B, N))
    output_action_log = jnp.zeros_like(output_action)

    for i in range(N):
        logit = decoder(shifted_action, obs_rep)[:, i, :]  # (B, A)
        masked_logits = jnp.where(
            legal_actions[:, i, :],
            logit,
            jnp.finfo(jnp.float32).min,
        )
        key, sample_key = jax.random.split(key)

        distribution = IdentityTransformation(distribution=tfd.Categorical(logits=masked_logits))
        action = distribution.sample(seed=sample_key)  # (B, )
        action_log = distribution.log_prob(action)  # (B, )

        output_action = output_action.at[:, i].set(action)
        output_action_log = output_action_log.at[:, i].set(action_log)

        # Adds all except the last action to shifted_actions, as it is out of range
        shifted_action = shifted_action.at[:, i + 1, 1:].set(
            jax.nn.one_hot(action, action_dim), mode="drop"
        )

    return output_action.astype(jnp.int32), output_action_log  # (B, N), (B, N)


def continuous_autoregressive_act(
    decoder: nn.Module,
    obs_rep: chex.Array,  # (B, N, E)
    action_dim: int,  # (, )
    legal_actions: Union[chex.Array, None],
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array]:
    # We don't need legal_actions for continuous actions but keep it to keep the APIs consistent.
    del legal_actions
    B, N, _ = obs_rep.shape
    shifted_action = jnp.zeros((B, N, action_dim))
    output_action = jnp.zeros((B, N, action_dim))
    output_action_log = jnp.zeros((B, N))

    for i in range(N):
        act_mean = decoder(shifted_action, obs_rep)[:, i, :]  # (B, A)
        action_std = jax.nn.softplus(decoder.log_std)

        key, sample_key = jax.random.split(key)

        distribution = tfd.Normal(loc=act_mean, scale=action_std)
        distribution = tfd.Independent(
            TanhTransformedDistribution(distribution),
            reinterpreted_batch_ndims=1,
        )
        action = distribution.sample(seed=sample_key)  # (B, A)
        action_log = distribution.log_prob(action)  # (B,)

        output_action = output_action.at[:, i, :].set(action)
        output_action_log = output_action_log.at[:, i].set(action_log)

        # Adds all except the last action to shifted_actions, as it is out of range
        shifted_action = shifted_action.at[:, i + 1, :].set(action, mode="drop")

    return output_action, output_action_log  # (B, N, A), (B, N)
