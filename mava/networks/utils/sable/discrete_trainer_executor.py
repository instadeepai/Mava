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

from typing import Callable, Optional, Tuple

import chex
import distrax
import jax
import jax.numpy as jnp
from flax import linen as nn

from mava.systems.sable.types import HiddenStates


def train_encoder_parallel(
    encoder: nn.Module,
    obs: chex.Array,
    hstate: chex.Array,
    dones: chex.Array,
    timestep_id: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Parallel encoding for discrete action spaces."""
    # Apply the encoder
    v_loc, obs_rep, updated_hstate = encoder(obs, hstate, dones, timestep_id)
    return v_loc, obs_rep, updated_hstate


def train_encoder_chunkwise(
    encoder: nn.Module,
    obs: chex.Array,
    hstate: chex.Array,
    dones: chex.Array,
    timestep_id: chex.Array,
    chunk_size: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Chunkwise encoding for discrete action spaces."""
    # Get the batch and sequence dimensions
    batch_dim, seq_dim = obs.shape[:2]
    # Initialize the value location and observation representation
    v_loc = jnp.zeros((batch_dim, seq_dim, 1))
    obs_rep = jnp.zeros((batch_dim, seq_dim, encoder.embed_dim))

    # Apply the encoder per chunk
    num_chunks = seq_dim // chunk_size
    for chunk_id in range(0, num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = (chunk_id + 1) * chunk_size
        chunk_obs = obs[:, start_idx:end_idx]
        chunk_dones = dones[:, start_idx:end_idx]
        chunk_timestep_id = timestep_id[:, start_idx:end_idx]
        # Apply parallel encoding per chunk
        chunk_v_loc, chunk_obs_rep, hstate = train_encoder_parallel(
            encoder=encoder,
            obs=chunk_obs,
            hstate=hstate,
            dones=chunk_dones,
            timestep_id=chunk_timestep_id,
        )
        v_loc = v_loc.at[:, start_idx:end_idx].set(chunk_v_loc)
        obs_rep = obs_rep.at[:, start_idx:end_idx].set(chunk_obs_rep)

    return v_loc, obs_rep, hstate


def train_decoder_fn(
    act_fn: Callable,
    decoder: nn.Module,
    obs_rep: chex.Array,
    action: chex.Array,
    legal_actions: chex.Array,
    hstates: chex.Array,
    dones: chex.Array,
    timestep_id: chex.Array,
    n_agents: int,
    rng_key: Optional[chex.PRNGKey] = None,
) -> Tuple[chex.Array, chex.Array]:
    """Parallel action sampling for discrete action spaces."""
    # Delete `rng_key` since it is not used in discrete action space
    del rng_key

    # Get the shifted actions for predicting the next action
    shifted_actions = get_shifted_actions(action, legal_actions, n_agents=n_agents)

    logit, _ = act_fn(
        decoder=decoder,
        obs_rep=obs_rep,
        shifted_actions=shifted_actions,
        hstates=hstates,
        dones=dones,
        timestep_id=timestep_id,
        legal_actions=legal_actions,
    )

    # Mask the logits for illegal actions
    masked_logits = jnp.where(
        legal_actions,
        logit,
        jnp.finfo(jnp.float32).min,
    )

    # Create a categorical distribution over the masked logits
    distribution = distrax.Categorical(logits=masked_logits)

    # Compute the log probability of the actions
    action_log_prob = distribution.log_prob(action)
    action_log_prob = jnp.expand_dims(action_log_prob, axis=-1)

    # Compute the entropy of the action distribution
    entropy = jnp.expand_dims(distribution.entropy(), axis=-1)

    return action_log_prob, entropy


def act_parallel(
    decoder: nn.Module,
    obs_rep: chex.Array,
    shifted_actions: chex.Array,
    hstates: Tuple[chex.Array, chex.Array],
    dones: chex.Array,
    timestep_id: chex.Array,
    legal_actions: chex.Array,
) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
    del legal_actions
    # Apply the decoder
    logit, updated_hstates = decoder(
        action=shifted_actions,
        obs_rep=obs_rep,
        hstates=hstates,
        dones=dones,
        timestep_id=timestep_id,
    )
    return logit, updated_hstates


def act_chunkwise(
    decoder: nn.Module,
    obs_rep: chex.Array,
    shifted_actions: chex.Array,
    hstates: Tuple[chex.Array, chex.Array],
    dones: chex.Array,
    timestep_id: chex.Array,
    legal_actions: chex.Array,
    chunk_size: int,
) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
    logit = jnp.zeros_like(legal_actions, dtype=jnp.float32)

    # Apply the decoder per chunk
    num_chunks = shifted_actions.shape[1] // chunk_size
    for chunk_id in range(0, num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = (chunk_id + 1) * chunk_size
        chunked_obs_rep = obs_rep[:, start_idx:end_idx]
        chunk_shifted_actions = shifted_actions[:, start_idx:end_idx]
        chunk_dones = dones[:, start_idx:end_idx]
        chunk_timestep_id = timestep_id[:, start_idx:end_idx]
        # Apply parallel encoding per chunk
        chunk_logit, hstates = act_parallel(
            decoder=decoder,
            obs_rep=chunked_obs_rep,
            shifted_actions=chunk_shifted_actions,
            hstates=hstates,
            dones=chunk_dones,
            timestep_id=chunk_timestep_id,
            legal_actions=legal_actions,
        )
        logit = logit.at[:, start_idx:end_idx].set(chunk_logit)

    return logit, hstates


def get_shifted_actions(action: chex.Array, legal_actions: chex.Array, n_agents: int) -> chex.Array:
    """Get the shifted action sequence for predicting the next action."""
    # Get the batch size, sequence length, and action dimension
    batch_size, sequence_size, action_dim = legal_actions.shape

    # Create a shifted action sequence for predicting the next action
    # Initialize the shifted action sequence.
    shifted_actions = jnp.zeros((batch_size, sequence_size, action_dim + 1))

    # Set the start-of-timestep token (first action as a "start" signal)
    start_timestep_token = jnp.zeros(action_dim + 1).at[0].set(1)

    # One hot encode the action
    one_hot_action = jax.nn.one_hot(action, action_dim)

    # Insert one-hot encoded actions into shifted array, shifting by 1 position
    shifted_actions = shifted_actions.at[:, :, 1:].set(one_hot_action)
    shifted_actions = jnp.roll(shifted_actions, shift=1, axis=1)

    # Set the start token for the first agent in each timestep
    shifted_actions = shifted_actions.at[:, ::n_agents, :].set(start_timestep_token)

    return shifted_actions


def init_sable(
    encoder: nn.Module,
    decoder: nn.Module,
    obs_carry: chex.Array,
    hstates: chex.Array,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, chex.Array, chex.Array, HiddenStates]:
    """Initializating the network: Applying non chunkwise encoding-decoding."""
    # Get the observation, legal actions, and timestep id
    obs, legal_actions, timestep_id = (
        obs_carry.agents_view,
        obs_carry.action_mask,
        obs_carry.step_count,
    )

    # Apply the encoder
    v_loc, obs_rep, _ = execute_encoder_parallel(
        encoder=encoder, obs=obs, decayed_hstate=hstates[0], timestep_id=timestep_id
    )

    # Apply the decoder
    _ = autoregressive_act(
        decoder=decoder,
        obs_rep=obs_rep,
        legal_actions=legal_actions,
        hstates=hstates[1],
        timestep_id=timestep_id,
        key=key,
    )

    return v_loc


def execute_encoder_parallel(
    encoder: nn.Module,
    obs: chex.Array,
    decayed_hstate: chex.Array,
    timestep_id: chex.Array,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Parallel encoding for discrete action spaces."""
    # Apply the encoder
    v_loc, obs_rep, updated_hstate = encoder.recurrent(obs, decayed_hstate, timestep_id)
    return v_loc, obs_rep, updated_hstate


def execute_encoder_chunkwise(
    encoder: nn.Module,
    obs: chex.Array,
    decayed_hstate: chex.Array,
    timestep_id: chex.Array,
    chunk_size: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Chunkwise encoding for Non-memory Sable and for discrete action spaces."""
    # Get the batch and sequence dimensions
    batch_dim, agents_per_seq = obs.shape[:2]
    # Initialize the value location and observation representation
    v_loc = jnp.zeros((batch_dim, agents_per_seq, 1))
    obs_rep = jnp.zeros((batch_dim, agents_per_seq, encoder.embed_dim))
    # Apply the encoder per chunk
    num_chunks = agents_per_seq // chunk_size
    for chunk_id in range(0, num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = (chunk_id + 1) * chunk_size
        chunk_obs = obs[:, start_idx:end_idx]
        chunk_timestep_id = timestep_id[:, start_idx:end_idx]
        # Apply parallel encoding per chunk
        chunk_v_loc, chunk_obs_rep, decayed_hstate = execute_encoder_parallel(
            encoder=encoder,
            obs=chunk_obs,
            decayed_hstate=decayed_hstate,
            timestep_id=chunk_timestep_id,
        )
        v_loc = v_loc.at[:, start_idx:end_idx].set(chunk_v_loc)
        obs_rep = obs_rep.at[:, start_idx:end_idx].set(chunk_obs_rep)

    return v_loc, obs_rep, decayed_hstate


def autoregressive_act(
    decoder: nn.Module,
    obs_rep: chex.Array,
    hstates: chex.Array,
    legal_actions: chex.Array,
    timestep_id: chex.Array,
    key: chex.PRNGKey,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    # Get the batch size, sequence length, and action dimension
    batch_size, n_agents, action_dim = legal_actions.shape

    # Create a shifted action sequence for predicting the next action
    # Initialize the shifted action sequence.
    shifted_actions = jnp.zeros((batch_size, n_agents, action_dim + 1))
    # Set the start-of-timestep token (first action as a "start" signal)
    shifted_actions = shifted_actions.at[:, 0, 0].set(1)

    # Define the output action and output action log sizes
    output_action = jnp.zeros((batch_size, n_agents, 1))
    output_action_log = jnp.zeros_like(output_action)

    # Apply the decoder autoregressively
    for i in range(n_agents):
        logit, updated_hstates = decoder.recurrent(
            action=shifted_actions[:, i : i + 1, :],
            obs_rep=obs_rep[:, i : i + 1, :],
            hstates=hstates,
            timestep_id=timestep_id[:, i : i + 1],
        )
        # Mask the logits for illegal actions
        masked_logits = jnp.where(
            legal_actions[:, i : i + 1, :],
            logit,
            jnp.finfo(jnp.float32).min,
        )
        # Create a categorical distribution over the masked logits
        distribution = distrax.Categorical(logits=masked_logits)
        # Sample an action from the distribution
        key, sample_key = jax.random.split(key)
        action, action_log = distribution.sample_and_log_prob(seed=sample_key)
        # Set the action and action log
        output_action = output_action.at[:, i, :].set(action)
        output_action_log = output_action_log.at[:, i, :].set(action_log)

        # Update the shifted action
        update_shifted_action = i + 1 < n_agents
        shifted_actions = jax.lax.cond(
            update_shifted_action,
            lambda action=action, i=i, shifted_actions=shifted_actions: shifted_actions.at[
                :, i + 1, 1:
            ].set(jax.nn.one_hot(action[:, 0], action_dim)),
            lambda shifted_actions=shifted_actions: shifted_actions,
        )

    return output_action.astype(jnp.int32), output_action_log, updated_hstates
