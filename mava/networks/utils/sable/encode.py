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

from typing import Tuple

import chex
import jax.numpy as jnp
from flax import linen as nn

# General shapes legend:
# B: batch size
# S: sequence length
# C: number of agents per chunk of sequence


def train_encoder_fn(
    encoder: nn.Module,
    obs: chex.Array,
    hstate: chex.Array,
    dones: chex.Array,
    step_count: chex.Array,
    chunk_size: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Chunkwise encoding for discrete action spaces."""
    B, S = obs.shape[:2]
    v_loc = jnp.zeros((B, S, 1))
    obs_rep = jnp.zeros((B, S, encoder.net_config.embed_dim))

    # Apply the encoder per chunk
    num_chunks = S // chunk_size
    for chunk_id in range(0, num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = (chunk_id + 1) * chunk_size
        # Chunk obs, dones, and step_count
        chunk_obs = obs[:, start_idx:end_idx]
        chunk_dones = dones[:, start_idx:end_idx]
        chunk_step_count = step_count[:, start_idx:end_idx]
        chunk_v_loc, chunk_obs_rep, hstate = encoder(
            chunk_obs, hstate, chunk_dones, chunk_step_count
        )
        v_loc = v_loc.at[:, start_idx:end_idx].set(chunk_v_loc)
        obs_rep = obs_rep.at[:, start_idx:end_idx].set(chunk_obs_rep)

    return v_loc, obs_rep, hstate


def act_encoder_fn(
    encoder: nn.Module,
    obs: chex.Array,
    decayed_hstate: chex.Array,
    step_count: chex.Array,
    chunk_size: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Chunkwise encoding for ff-Sable and for discrete action spaces."""
    B, C = obs.shape[:2]
    v_loc = jnp.zeros((B, C, 1))
    obs_rep = jnp.zeros((B, C, encoder.net_config.embed_dim))

    # Apply the encoder per chunk
    num_chunks = C // chunk_size
    for chunk_id in range(0, num_chunks):
        start_idx = chunk_id * chunk_size
        end_idx = (chunk_id + 1) * chunk_size
        # Chunk obs and step_count
        chunk_obs = obs[:, start_idx:end_idx]
        chunk_step_count = step_count[:, start_idx:end_idx]
        chunk_v_loc, chunk_obs_rep, decayed_hstate = encoder.recurrent(
            chunk_obs, decayed_hstate, chunk_step_count
        )
        v_loc = v_loc.at[:, start_idx:end_idx].set(chunk_v_loc)
        obs_rep = obs_rep.at[:, start_idx:end_idx].set(chunk_obs_rep)

    return v_loc, obs_rep, decayed_hstate
