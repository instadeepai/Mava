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
from flax import linen as nn

# General shapes legend:
# B: batch size
# S: sequence length
# C: number of agents per chunk of sequence


def train_encoder_fn(
    encoder: nn.Module,
    obs: chex.Array,
    hstate: chex.Array,
    scale: chex.Array,
    dones: chex.Array,
    step_count: chex.Array,
    chunk_size: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Chunkwise encoding for discrete action spaces."""
    B, S = obs.shape[:2]

    # Apply the encoder per chunk
    num_chunks = S // chunk_size
    v_loc, obs_rep, hstate, _ = encoder(obs, hstate, scale, dones, step_count, num_chunks)

    return v_loc, obs_rep, hstate


def act_encoder_fn(
    encoder: nn.Module,
    obs: chex.Array,
    decayed_hstate: chex.Array,
    scale: chex.Array,
    step_count: chex.Array,
    chunk_size: int,
) -> Tuple[chex.Array, chex.Array, chex.Array]:
    """Chunkwise encoding for ff-Sable and for discrete action spaces."""
    B, C = obs.shape[:2]

    # Apply the encoder per chunk
    v_loc, obs_rep, decayed_hstate = encoder.recurrent(
        obs,
        decayed_hstate,
        scale,
        step_count,
    )

    return v_loc, obs_rep, decayed_hstate
