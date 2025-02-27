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
from einops import rearrange
from omegaconf import DictConfig

from mava.networks.utils.sable import PositionalEncoding

# General shapes legend:
# B: batch size
# N: number of agents
# S: sequence length
# C: chunk size - T * N in a chunk
# T: number of timesteps


def get_decay_matrix(dones: Array, n_agents: int, masked: bool, decay_kappa: float) -> Array:
    """Get the decay matrix for the full sequence based on the dones and retention type."""
    # Extract done information at the timestep level
    timestep_dones = dones[:, ::n_agents]  # B, T

    # B, T, T
    timestep_mask = _get_decay_matrix_mask_timestep(timestep_dones)
    decay_matrix = _get_default_decay_matrix(timestep_dones, decay_kappa)
    decay_matrix *= timestep_mask

    # B, T, T ->  B, T * N, T * N
    decay_matrix = jnp.repeat(jnp.repeat(decay_matrix, n_agents, axis=1), n_agents, axis=2)

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
    timestep_dones = dones[:, ::n_agents]
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
    Compute decay matrices, xi, and delta arrays for multi-head inputs with an added chunk
        dimension.

    Args:
        dones: Dones array of shape (B, nC, C) where C = T * n_agents and nC is the number of
            chunks.
        decay_kappas: Array of decay kappas (one per head), shape (n_head,).
        n_agents: Number of agents.
        masked: Whether to apply causal masking.

    Returns:
        decay_matrix: Array of shape (B, nC, n_head, C, C)
        xi: Array of shape (B, nC, n_head, C, 1)
        chunk_decay: Array of shape (nC, n_head) computed as exp(decay_kappa * T) per head
            for each chunk.
        delta: Array of shape (B, nC, n_head, 1, 1) indicating per-head done mask.
    """
    B, nc, C = dones.shape
    n_head = decay_kappas.shape[0]
    # Number of timesteps per chunk.
    T = C // n_agents

    all_decay_matrices = []
    all_xis = []
    all_chunk_decays = []
    all_deltas = []

    for h in range(n_head):
        head_decay_kappa = decay_kappas[h]
        head_decay_matrices = []
        head_xis = []
        head_chunk_decays = []
        head_deltas = []
        head_chunk_decay = jnp.exp(head_decay_kappa * T)
        for c in range(nc):
            chunk_dones = dones[:, c, :]

            # Save the same chunk decay for each chunk.
            head_chunk_decays.append(head_chunk_decay)

            # Compute the decay matrix for this head and chunk.
            dm = get_decay_matrix(
                chunk_dones, n_agents, masked, head_decay_kappa
            )  # shape: (B, C, C)
            head_decay_matrices.append(dm)

            # Compute xi for this chunk.
            xi_chunk = get_xi(chunk_dones, n_agents, head_decay_kappa)  # shape: (B, C, 1)
            head_xis.append(xi_chunk)

            # Compute delta for this chunk.
            delta_h = ~jnp.any(chunk_dones[:, ::n_agents], axis=1)[
                :, jnp.newaxis, jnp.newaxis
            ]  # shape: (B, 1, 1)
            head_deltas.append(delta_h)

        # Stack over chunks:
        head_decay_chunks = jnp.stack(head_decay_matrices, axis=1)  # (B, nC, C, C)
        head_xi_chunks = jnp.stack(head_xis, axis=1)  # (B, nC, C, 1)
        head_delta = jnp.stack(head_deltas, axis=1)  # (B, nC, 1, 1)
        head_chunk_decays = jnp.stack(head_chunk_decays, axis=0)  # (nC,)

        all_decay_matrices.append(head_decay_chunks)
        all_xis.append(head_xi_chunks)
        all_chunk_decays.append(head_chunk_decays)
        all_deltas.append(head_delta)

    # Stack over head dimension:
    decay_matrix = jnp.stack(all_decay_matrices, axis=2)  # (B, nC, n_head, C, C)
    xi = jnp.stack(all_xis, axis=2)  # (B, nC, n_head, C, 1)
    chunk_decay = jnp.stack(all_chunk_decays, axis=1)  # (nC, n_head)
    delta = jnp.stack(all_deltas, axis=2)  # (B, nC, n_head, 1, 1)

    return decay_matrix, xi, chunk_decay, delta


def reshape_qkv(
    q_proj: Array, k_proj: Array, v_proj: Array, n_head: int, num_chunks: int
) -> Tuple[Array, Array, Array]:
    # missing the case where n_head = 1 and num_chunks = 1

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

    elif num_chunks == 1 and n_head == 1:
        q_proj = rearrange(q_proj, "B S n -> B () () S n")
        k_proj = rearrange(k_proj, "B S n -> B () () S n")
        v_proj = rearrange(v_proj, "B S n -> B () () S n")

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
        self.decay_kappas = self.decay_kappas * self.decay_scaling_factor
        self.decay_kappas = jnp.log(self.decay_kappas)

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
        self.w_k = self.param(
            "w_k",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.embed_dim),
        )
        self.w_q = self.param(
            "w_q",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.embed_dim),
        )
        self.w_v = self.param(
            "w_v",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.embed_dim),
        )
        self.group_norm = nn.GroupNorm(num_groups=self.n_head)

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
    ) -> Tuple[Array, Array]:
        """Chunkwise (default) representation of the multi-scale retention mechanism"""
        B, C, _ = value.shape

        # Positional encoding of the current step
        if self.memory_config.timestep_positional_encoding:
            key, query, value = self.pe(key, query, value, step_count)

        q_proj = query @ self.w_q
        k_proj = key @ self.w_k
        v_proj = value @ self.w_v

        # (B, num_chunks, num_heads, chunk_size, head_size)
        _q_proj, _k_proj, _v_proj = reshape_qkv(q_proj, k_proj, v_proj, self.n_head, num_chunks)
        _k_proj = _k_proj.transpose(0, 1, 2, -1, -2)

        dones = reshape_dones(dones, num_chunks)

        _decay_matrix, _xi, _chunk_decay, _delta = get_decay_matrices(
            dones, self.decay_kappas, self.n_agents, self.masked
        )

        ret_outputs = []
        for chunk in range(num_chunks):
            decay_matrix = _decay_matrix[:, chunk]
            xi = _xi[:, chunk]
            chunk_decay = _chunk_decay[chunk]
            delta = _delta[:, chunk]
            q_proj = _q_proj[:, chunk]
            k_proj = _k_proj[:, chunk]
            v_proj = _v_proj[:, chunk]

            if self.memory_config.type == "ff_sable":
                decay_matrix = jnp.ones_like(decay_matrix)
                decay_matrix = jax.vmap(_causal_mask, in_axes=(1, None), out_axes=1)(
                    decay_matrix, self.masked
                )
                xi = jnp.ones_like(xi)
                next_hstate = (k_proj @ v_proj) + hstate
                del chunk_decay, delta
            else:
                next_hstate = (
                    k_proj @ (v_proj * decay_matrix[:, :, -1].reshape(B, self.n_head, C, 1))
                    + hstate * chunk_decay[None, :, None, None] * delta
                )

            cross_chunk = (q_proj @ hstate) * xi
            inner_chunk = ((q_proj @ k_proj) * decay_matrix) @ v_proj

            hstate = next_hstate

            ret_output = cross_chunk + inner_chunk
            ret_outputs.append(ret_output)

        ret_output = jnp.stack(ret_outputs, axis=1)
        ret_output = rearrange(ret_output, "B nC nh Cs hs -> B nh (nC Cs) hs", nh=self.n_head)

        # Joint heads again
        ret_output = rearrange(ret_output, "B nh C hs -> B C (nh hs)", nh=self.n_head)
        ret_output = self.group_norm(ret_output.reshape(-1, self.head_size)).reshape(
            (B, C, self.embed_dim)
        )

        x = key
        output = (jax.nn.swish(x @ self.w_g) * ret_output) @ self.w_o
        return output, hstate

    def recurrent(
        self, key_n: Array, query_n: Array, value_n: Array, hstate: Array, step_count: Array
    ) -> Tuple[Array, Array]:
        """Recurrent representation of the multi-scale retention mechanism"""

        B, S, _ = value_n.shape

        # Positional encoding of the current step if enabled
        if self.memory_config.timestep_positional_encoding:
            key_n, query_n, value_n = self.pe(key_n, query_n, value_n, step_count)

        q_proj = query_n @ self.w_q
        k_proj = key_n @ self.w_k
        v_proj = value_n @ self.w_v

        # Never chunk in the recurrent case
        q_proj, k_proj, v_proj = q_proj, k_proj, v_proj = reshape_qkv(
            q_proj, k_proj, v_proj, self.n_head, 1
        )
        k_proj = k_proj.transpose(0, 1, 2, -1, -2)

        # Remove chunk dim
        q_proj, k_proj, v_proj = q_proj.squeeze(1), k_proj.squeeze(1), v_proj.squeeze(1)

        updated_hstate = hstate + (k_proj @ v_proj)
        ret_output = q_proj @ updated_hstate

        # Joint heads again
        ret_output = rearrange(ret_output, "B nh C hs -> B C (nh hs)", nh=self.n_head)

        ret_output = self.group_norm(ret_output.reshape(-1, self.head_size)).reshape(
            (B, S, self.embed_dim)
        )

        x = key_n
        output = (jax.nn.swish(x @ self.w_g) * ret_output) @ self.w_o
        return output, updated_hstate
