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

from typing import Optional, Tuple

import chex
import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn

# General shapes legend:
# B: batch size
# S: sequence length
# A: number of actions
# N: number of agents


def train_decoder_fn(
    decoder: nn.Module,
    obs_rep: chex.Array,
    action: chex.Array,
    legal_actions: chex.Array,
    hstates: chex.Array,
    dones: chex.Array,
    step_count: chex.Array,
    n_agents: int,
    chunk_size: int,
    rng_key: Optional[chex.PRNGKey] = None,
) -> Tuple[chex.Array, chex.Array]:
    """Parallel action sampling for discrete action spaces."""
    # Delete `rng_key` since it is not used in discrete action space
    del rng_key

    shifted_actions = get_shifted_actions(action, legal_actions, n_agents=n_agents)
    logit = jnp.zeros_like(legal_actions, dtype=jnp.float32)

    # Apply the decoder per chunk
    num_chunks = shifted_actions.shape[1] // chunk_size
    for chunk_id in range(0, num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = (chunk_id + 1) * chunk_size
        # Chunk obs_rep, shifted_actions, dones, and step_count
        chunked_obs_rep = obs_rep[:, start_idx:end_idx]
        chunk_shifted_actions = shifted_actions[:, start_idx:end_idx]
        chunk_dones = dones[:, start_idx:end_idx]
        chunk_step_count = step_count[:, start_idx:end_idx]
        chunk_logit, hstates = decoder(
            action=chunk_shifted_actions,
            obs_rep=chunked_obs_rep,
            hstates=hstates,
            dones=chunk_dones,
            step_count=chunk_step_count,
        )
        logit = logit.at[:, start_idx:end_idx].set(chunk_logit)

    masked_logits = jnp.where(
        legal_actions,
        logit,
        jnp.finfo(jnp.float32).min,
    )

    distribution = distrax.Categorical(logits=masked_logits)
    action_log_prob = distribution.log_prob(action)
    action_log_prob = jnp.expand_dims(action_log_prob, axis=-1)
    entropy = jnp.expand_dims(distribution.entropy(), axis=-1)

    return action_log_prob, entropy


def get_shifted_actions(action: chex.Array, legal_actions: chex.Array, n_agents: int) -> chex.Array:
    """Get the shifted action sequence for predicting the next action."""
    B, S, A = legal_actions.shape

    # Create a shifted action sequence for predicting the next action
    shifted_actions = jnp.zeros((B, S, A + 1))

    # Set the start-of-timestep token (first action as a "start" signal)
    start_timestep_token = jnp.zeros(A + 1).at[0].set(1)

    # One hot encode the action
    one_hot_action = jax.nn.one_hot(action, A)

    # Insert one-hot encoded actions into shifted array, shifting by 1 position
    shifted_actions = shifted_actions.at[:, :, 1:].set(one_hot_action)
    shifted_actions = jnp.roll(shifted_actions, shift=1, axis=1)

    # Set the start token for the first agent in each timestep
    shifted_actions = shifted_actions.at[:, ::n_agents, :].set(start_timestep_token)

    return shifted_actions


def autoregressive_act(
    decoder: nn.Module,
    obs_rep: chex.Array,
    hstates: chex.Array,
    legal_actions: chex.Array,
    step_count: chex.Array,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    B, N, A = legal_actions.shape

    shifted_actions = jnp.zeros((B, N, A + 1))
    shifted_actions = shifted_actions.at[:, 0, 0].set(1)

    output_action = jnp.zeros((B, N, 1))
    output_action_log = jnp.zeros_like(output_action)

    # Apply the decoder autoregressively
    for i in range(N):
        logit, hstates = decoder.recurrent(
            action=shifted_actions[:, i : i + 1, :],
            obs_rep=obs_rep[:, i : i + 1, :],
            hstates=hstates,
            step_count=step_count[:, i : i + 1],
        )
        masked_logits = jnp.where(
            legal_actions[:, i : i + 1, :],
            logit,
            jnp.finfo(jnp.float32).min,
        )
        distribution = distrax.Categorical(logits=masked_logits)
        key, sample_key = jax.random.split(key)
        action, action_log = distribution.sample_and_log_prob(seed=sample_key)
        output_action = output_action.at[:, i, :].set(action)
        output_action_log = output_action_log.at[:, i, :].set(action_log)

        # Adds all except the last action to shifted_actions, as it is out of range.
        shifted_actions = shifted_actions.at[:, i + 1, 1:].set(
            jax.nn.one_hot(action[:, 0], A), mode="drop"
        )

    return output_action.astype(jnp.int32), output_action_log, hstates
