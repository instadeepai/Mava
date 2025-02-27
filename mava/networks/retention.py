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

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Array
from omegaconf import DictConfig

from mava.networks.utils.sable import PositionalEncoding
from einops import rearrange

# General shapes legend:
# B: batch size
# N: number of agents
# S: sequence length
# C: chunk size - T * N in a chunk
# T: number of timesteps

def get_decay_matrix(dones: Array, n_agents: int, masked: bool, decay_kappa: float) -> Array:
    """Get the decay matrix for the full sequence based on the dones and retention type."""
    # Extract done information at the timestep level
    timestep_dones = dones[:, :: n_agents]  # B, T

    # B, T, T
    timestep_mask = _get_decay_matrix_mask_timestep(timestep_dones)
    decay_matrix = _get_default_decay_matrix(timestep_dones, decay_kappa)
    decay_matrix *= timestep_mask

    # B, T, T ->  B, T * N, T * N
    decay_matrix = jnp.repeat(
        jnp.repeat(decay_matrix, n_agents, axis=1), n_agents, axis=2
    )

    # Apply a causal mask over agents if full self-retention is disabled
    # This converts it from a blocked decay matrix to a causal decay matrix
    decay_matrix = _causal_mask(decay_matrix, masked)

    return decay_matrix

def _causal_mask(matrix: Array, masked: bool) -> Array:
    """Applies a causal mask to the input matrix if `masked` is True."""
    if masked:
        mask_agents = jnp.tril(jnp.ones((matrix.shape[1], matrix.shape[1])))
        matrix = mask_agents[None, :, :] * matrix
    return matrix

def _get_decay_matrix_mask_timestep(ts_dones: Array) -> Array:
    """Generates a mask over the timesteps based on the done status of agents.

    If there is a termination on timestep t, then the decay matrix should be
    restarted from index (t, t). See the section Adapting the decay matrix for MARL
    for a full explanation: https://arxiv.org/pdf/2410.01706
    """
    # Get the shape of the input: batch size and number of timesteps
    B, T = ts_dones.shape

    # Initialise the mask
    timestep_mask = jnp.zeros((B, T, T), dtype=bool)
    all_false = jnp.zeros((B, T, T), dtype=bool)

    # Iterate over the timesteps and apply the mask
    for i in range(T):
        done_this_step = ts_dones[:, i, jnp.newaxis, jnp.newaxis]
        ts_done_xs = all_false.at[:, i:, :].set(done_this_step)
        ts_done_ys = all_false.at[:, :, :i].set(done_this_step)

        # Combine the x and y masks to get the mask for the current timestep.
        timestep_mask |= ts_done_xs & ts_done_ys

    return ~timestep_mask

def _get_default_decay_matrix(dones: Array, decay_kappa: float) -> Array:
    """Compute the decay matrix without taking into account the timestep-based masking."""
    # Get the shape of the input: batch size and number of timesteps
    B, T = dones.shape

    # Create the n and m matrices
    n = jnp.arange(T)[:, jnp.newaxis, ...]
    m = jnp.arange(T)[jnp.newaxis, ...]

    # Decay based on difference in timestep indices.
    decay_matrix = (decay_kappa * (n - m)) * (n >= m)
    decay_matrix = jnp.exp(decay_matrix)

    # Zero out upper-triangular values (excluding the main diagonal)
    decay_matrix = decay_matrix * jnp.tril(jnp.ones((T, T)))

    # Adjust for batch size
    decay_matrix = jnp.broadcast_to(decay_matrix, (B, T, T))

    return decay_matrix

def get_xi(dones: Array, n_agents: int, decay_kappa: float) -> Array:
    """Computes a decaying matrix 'xi', which decays over time until the first done signal."""
    # Get done status for each timestep by slicing out the agent dimension
    timestep_dones = dones[:, :: n_agents]
    B, T = timestep_dones.shape

    # Compute the first done step for each sequence,
    # or set it to sequence length if no dones exist
    first_dones = jnp.where(
        ~jnp.any(timestep_dones, axis=1, keepdims=True),
        jnp.full((B, 1), T),
        jnp.argmax(timestep_dones, axis=1, keepdims=True),
    )

    xi = jnp.zeros((B, T, 1))
    # Fill 'xi' with decaying values up until the first done step
    for i in range(T):
        before_first_done = i < first_dones
        xi_i = jnp.exp(decay_kappa * (i + 1)) * before_first_done
        xi = xi.at[:, i, :].set(xi_i)

    # Repeat the decay matrix 'xi' for all agents
    xi = jnp.repeat(xi, n_agents, axis=1)

    return xi

def get_decay_matrices(
    dones: Array,
    decay_kappas: Array,  # shape: (n_head,)
    n_agents: int,
    masked: bool,
) -> Tuple[Array, Array, Array, Array]:
    """
    Compute decay matrices, xi, chunk_decay, and delta arrays for multi-head and chunked inputs.

    Args:
        dones: Reshaped dones of shape (B, num_chunks, C) where C = T * n_agents.
        decay_kappas: Array of decay kappas (one per head), shape (n_head,).
        n_agents: Number of agents.
        masked: Whether to apply causal masking.
        bsz: Batch size.
        num_chunks: Number of chunks.
    
    Returns:
        decay_matrix: Array of shape (B, num_chunks, n_head, C, C)
        xi: Array of shape (B, num_chunks, n_head, C, 1)
        chunk_decay: Array of shape (n_head,) computed as decay_kappa ** (C // n_agents) per head.
        delta: Array of shape (B, num_chunks, n_head, 1) indicating per-head, per-chunk done mask.
    """
    B, nC, C = dones.shape
    n_head = decay_kappas.shape[0]
    # In each chunk, the number of timesteps is:
    T = C // n_agents

    decay_matrices_list = []
    xi_list = []
    chunk_decay_list = []
    delta_list = []  # per head

    # Loop over heads: each head uses its own decay_kappa.
    for h in range(n_head):
        head_decay_kappa = decay_kappas[h]
        head_decay_chunks = []
        head_xi_chunks = []
        head_delta_chunks = []
        # Compute the scalar chunk_decay for this head (matching the SimpleRetention version)
        head_chunk_decay = jnp.exp(head_decay_kappa * T)
        for c in range(nC):
            # Extract the dones for this chunk (shape: (B, C))
            chunk_dones = dones[:, c, :]

            # Compute the decay matrix for this chunk using your helper function.
            dm = get_decay_matrix(chunk_dones, n_agents, masked, head_decay_kappa)  # shape: (B, C, C)
            head_decay_chunks.append(dm)

            # Compute xi for this chunk.
            xi_chunk = get_xi(chunk_dones, n_agents, head_decay_kappa)  # shape: (B, C, 1)
            head_xi_chunks.append(xi_chunk)

            # Compute delta for this head and chunk.
            # First, slice the done signals to get one per timestep: shape (B, T)
            timestep_dones = chunk_dones[:, ::n_agents]
            # delta is True (or 1) if no done occurred in the chunk.
            delta_chunk = ~jnp.any(timestep_dones, axis=1)  # shape: (B,)
            # Expand dims for consistency.
            delta_chunk = delta_chunk[:, None]  # shape: (B, 1)
            head_delta_chunks.append(delta_chunk)

        # Stack over chunks: shape (B, nC, C, C) for decay matrix,
        # (B, nC, C, 1) for xi, and (B, nC, 1) for delta.
        head_decay_chunks = jnp.stack(head_decay_chunks, axis=1)
        head_xi_chunks = jnp.stack(head_xi_chunks, axis=1)
        head_delta = jnp.stack(head_delta_chunks, axis=1)

        decay_matrices_list.append(head_decay_chunks)
        xi_list.append(head_xi_chunks)
        chunk_decay_list.append(head_chunk_decay)
        delta_list.append(head_delta)  # per head delta: (B, nC, 1)

    # Stack over head dimension:
    decay_matrix = jnp.stack(decay_matrices_list, axis=2)  # (B, nC, n_head, C, C)
    xi = jnp.stack(xi_list, axis=2)                      # (B, nC, n_head, C, 1)
    chunk_decay = jnp.stack(chunk_decay_list, axis=0)    # (n_head,)
    delta = jnp.stack(delta_list, axis=2)                # (B, nC, n_head, 1)

    return decay_matrix, xi, chunk_decay, delta



def reshape_qkv(q_proj: Array, k_proj: Array, v_proj: Array, n_head: int, num_chunks: int) -> Tuple[Array, Array, Array]:

    # split the embeddings over heads and the sequence over chunks
    if n_head > 1 and num_chunks > 1:
        q_proj = rearrange(q_proj, "B (Cs nC) (nh hs) -> B nC nh Cs hs", nh=n_head, nC=num_chunks)
        k_proj = rearrange(k_proj, "B (Cs nC) (nh hs) -> B nC nh Cs hs", nh=n_head, nC=num_chunks)
        v_proj = rearrange(v_proj, "B (Cs nC) (nh hs) -> B nC nh Cs hs", nh=n_head, nC=num_chunks)

    # add dummy chunk dim, split the embeddings over heads
    elif n_head > 1 and num_chunks == 1:
        q_proj = rearrange(q_proj, "B S (nh hs) -> B () nh S hs", nh=n_head)
        k_proj = rearrange(k_proj, "B S (nh hs) -> B () nh S hs", nh=n_head)
        v_proj = rearrange(v_proj, "B S (nh hs) -> B () nh S hs", nh=n_head)

    # add dummy head dim, split the embeddings over chunks
    elif num_chunks > 1 and n_head == 1:
        q_proj = rearrange(q_proj, "B (Cs nC) n -> B nC () Cs n", nC=num_chunks)
        k_proj = rearrange(k_proj, "B (Cs nC) n -> B nC () Cs n", nC=num_chunks)
        v_proj = rearrange(v_proj, "B (Cs nC) n -> B nC () Cs n", nC=num_chunks)
    
    return q_proj, k_proj, v_proj

def reshape_dones(dones: Array, num_chunks: int) -> Tuple[Array, Array]:

    # split sequence over chunks
    if num_chunks > 1:
        dones = rearrange(dones, "B (nC Cs) -> B nC Cs")

    # add a dummy chunk dim    
    else:
        dones = rearrange(dones, "B S -> B () S")
    
    return dones



class MultiScaleRetention(nn.Module):
    """Multi-scale retention mechanism for Sable."""

    embed_dim: int
    n_head: int
    n_agents: int
    memory_config: DictConfig
    masked: bool = True
    decay_scaling_factor: float = 1.0

    def setup(self) -> None:
        assert self.embed_dim % self.n_head == 0, "embed_dim must be divisible by n_head"
        self.head_size = self.embed_dim // self.n_head

        # Decay kappa for each head
        self.decay_kappas = 1 - jnp.exp(
            jnp.linspace(jnp.log(1 / 32), jnp.log(1 / 512), self.n_head)
        )
        self.decay_kappas = jnp.log(self.decay_kappas * self.decay_scaling_factor)

        # Initialise the weights and group norm
        self.w_g = self.param(
            "w_g",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.embed_dim),
        )
        self.w_o = self.param(
            "w_o",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.embed_dim),
        )
        self.group_norm = nn.GroupNorm(
            num_groups=self.n_head,
            group_size=None,
        )

        # Initialise the weights
        self.w_q = self.param(
            "w_q",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.embed_dim),
        )
        self.w_k = self.param(
            "w_k",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.embed_dim),
        )
        self.w_v = self.param(
            "w_v",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.embed_dim),
        )

        # Create an instance of the positional encoding
        self.pe = PositionalEncoding(self.embed_dim)

    def __call__(
        self,
        key: Array,
        query: Array,
        value: Array,
        hstate: Array,
        dones: Array,
        step_count: Array,
        num_chunks: int,
        scale: Array,

    ) -> Tuple[Array, Array]:
        """Chunkwise (default) representation of the multi-scale retention mechanism"""

        # Positional encoding of the current step
        if self.memory_config.timestep_positional_encoding:
            key, query, value = self.pe(key, query, value, step_count)

        q_proj = query @ self.w_q
        k_proj = key @ self.w_k
        v_proj = value @ self.w_v
        B, S = key.shape[:2]
        
        # (B, num_chunks, num_heads, chunk_size, head_size)
        q_proj, k_proj, v_proj = reshape_qkv(q_proj, k_proj, v_proj, self.n_head, num_chunks)
        k_proj_t = k_proj.transpose(0, 1, 2, 4, 3)
        # (B, num_chunks, chunk_size)
        dones = reshape_dones(dones, num_chunks)
        # just to a single chunk for now.

        decay_matrix, xi, chunk_decay, delta = get_decay_matrices(
            dones, self.decay_kappas, self.n_agents, self.masked
        )
        
        decay_matrix_scale = jnp.sqrt(decay_matrix.sum(axis=-1, keepdims=True))

        qk_mat = q_proj @ k_proj_t * (decay_matrix / decay_matrix_scale)
        inner_scale = jnp.clip(jnp.abs(qk_mat).sum(axis=-1, keepdims=True), min=1.0)
        qk_mat = qk_mat / inner_scale
        inner_output = qk_mat @ v_proj

        # reduce kv to one chunk
        # original mistake
        # value_inner_decay = decay_matrix[:, :, :, :, -1] / decay_matrix[:, :, :, :, -1].sum(axis=-1, keepdims=True)
        # select last row not last column now 
        value_inner_decay = decay_matrix[:, :, :, -1, :] / decay_matrix[:, :, :, -1, :].sum(axis=-1, keepdims=True)
        value_inner_decay = value_inner_decay[..., jnp.newaxis]
        kv = k_proj_t @ (v_proj * value_inner_decay)

        kv_recurrent = []
        cross_scale = []

        kv_scale = scale

        for i in range(num_chunks):
            kv_recurrent.append(hstate / kv_scale)
            cross_scale.append(kv_scale)
            hstate = hstate * chunk_decay[jnp.newaxis, :, jnp.newaxis, jnp.newaxis] * delta[:, i][..., jnp.newaxis] + kv[:, i] 
            kv_scale = jnp.clip(jnp.abs(hstate).sum(axis=-2, keepdims=True).max(axis=-1, keepdims=True), min=1.0)

        kv_recurrent = jnp.stack(kv_recurrent, axis=1)
        cross_scale = jnp.stack(cross_scale, axis=1)

        all_scale = jnp.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = (q_proj * xi) @ kv_recurrent

        ret_output = inner_output / align_inner_scale + cross_output / align_cross_scale
        
        ret_output = rearrange(ret_output, "B nC nh Cs n -> B (nC Cs) (nh n)", nh=self.n_head)
        ret_output = self.group_norm(ret_output.reshape(-1, self.head_size)).reshape(
            B, S, self.embed_dim
        )
        
        x = key
        output = (jax.nn.swish(x @ self.w_g) * ret_output) @ self.w_o
        return output, hstate

    def recurrent(
        self, key_n: Array, query_n: Array, value_n: Array, hstate: Array, step_count: Array, scale: Array,
    ) -> Tuple[Array, Array]:
        """Recurrent representation of the multi-scale retention mechanism"""

        # Positional encoding of the current step if enabled
        if self.memory_config.timestep_positional_encoding:
            key_n, query_n, value_n = self.pe(key_n, query_n, value_n, step_count)

        q_proj = query_n @ self.w_q
        k_proj = key_n @ self.w_k
        v_proj = value_n @ self.w_v
        B = key_n.shape[0]
        
        # Reshape to have heads dim
        # B: batch
        # S: sequence length
        # n: num heads
        # h: head size
        q_proj = rearrange(q_proj, "B S (nh hs) -> B nh S hs", nh=self.n_head)
        k_proj = rearrange(k_proj, "B S (nh hs) -> B nh S hs", nh=self.n_head)
        v_proj = rearrange(v_proj, "B S (nh hs) -> B nh hs S", nh=self.n_head)
        # v_proj = rearrange(v_proj, "B S (n h) -> B h S n", h=self.n_head)

        # updated_hstate = hstate + (k_proj.transpose(0, 1, -1, -2) @ v_proj) / jnp.sqrt(scale)
        # ret = q_proj @ updated_hstate
        
        kv = k_proj * v_proj
        updated_hstate = hstate + kv / jnp.sqrt(scale)
        ret = (q_proj * updated_hstate).sum(axis=-1)
        ret_output = self.group_norm(ret.reshape(-1, self.head_size)).reshape(
            B, 1, self.embed_dim
        )

        x = key_n
        output = (jax.nn.swish(x @ self.w_g) * ret_output) @ self.w_o
        return output, updated_hstate
