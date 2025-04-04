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

import copy
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
    decay_matrix_timesteps = _get_default_decay_matrix(timestep_dones, decay_kappa)
    decay_matrix_timesteps *= timestep_mask

    # B, T, T ->  B, T * N, T * N
    decay_matrix_broadcast_over_agents = jnp.repeat(
        jnp.repeat(decay_matrix_timesteps, n_agents, axis=1), n_agents, axis=2
    )

    # Apply a causal mask over agents if full self-retention is disabled
    # This converts it from a blocked decay matrix to a causal decay matrix
    decay_matrix_broadcast_over_agents = _causal_mask(decay_matrix_broadcast_over_agents, masked)

    return decay_matrix_broadcast_over_agents, decay_matrix_timesteps


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
    xi_timesteps = copy.deepcopy(xi)
    xi_broadcast_over_agents = jnp.repeat(xi, n_agents, axis=1)

    return xi_broadcast_over_agents, xi_timesteps


def get_decay_matrices(
    dones: Array,
    decay_kappas: Array,  # shape: (n_head,)
    n_agents: int,
    masked: bool,
) -> Tuple[Array, Array, Array, Array, Array, Array]:
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
    all_dm_times = []
    all_xi_times = []

    for h in range(n_head):
        head_decay_kappa = decay_kappas[h]
        head_decay_matrices = []
        head_xis = []
        head_chunk_decays = []
        head_deltas = []
        _head_dm_times = []
        _head_xi_times = []
        head_chunk_decay = jnp.exp(head_decay_kappa * T)
        for c in range(nc):
            chunk_dones = dones[:, c, :]

            # Save the same chunk decay for each chunk.
            head_chunk_decays.append(head_chunk_decay)

            # Compute the decay matrix for this head and chunk.
            dm, dm_timesteps = get_decay_matrix(
                chunk_dones, n_agents, masked, head_decay_kappa
            )  # shape: (B, C, C)
            head_decay_matrices.append(dm)
            _head_dm_times.append(dm_timesteps)

            # Compute xi for this chunk.
            xi_chunk, xi_chunk_timesteps = get_xi(
                chunk_dones, n_agents, head_decay_kappa
            )  # shape: (B, C, 1)
            head_xis.append(xi_chunk)
            _head_xi_times.append(xi_chunk_timesteps)

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
        head_xi_times = jnp.stack(_head_xi_times, axis=1)  # (B, nC, T, 1)
        head_dm_times = jnp.stack(_head_dm_times, axis=1)  # (B, nC, T, T)

        all_decay_matrices.append(head_decay_chunks)
        all_xis.append(head_xi_chunks)
        all_chunk_decays.append(head_chunk_decays)
        all_deltas.append(head_delta)
        all_xi_times.append(head_xi_times)
        all_dm_times.append(head_dm_times)

    # Stack over head dimension:
    decay_matrix = jnp.stack(all_decay_matrices, axis=2)  # (B, nC, n_head, C, C)
    xi = jnp.stack(all_xis, axis=2)  # (B, nC, n_head, C, 1)
    chunk_decay = jnp.stack(all_chunk_decays, axis=1)  # (nC, n_head)
    delta = jnp.stack(all_deltas, axis=2)  # (B, nC, n_head, 1, 1)
    decay_matrix_times = jnp.stack(all_dm_times, axis=2)  # (B, nC, n_head, T, T)
    xi_times = jnp.stack(all_xi_times, axis=2)  # (B, nC, n_head, T, 1)

    return decay_matrix, xi, chunk_decay, delta, decay_matrix_times, xi_times


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
        dones = rearrange(dones, "B (nC Cs) -> B nC Cs", nC=num_chunks)

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
        self.scaling = self.head_size**-0.5

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
        kv_scale: Array,
        inference: bool = False,
    ) -> Tuple[Array, Array, Array]:
        """Chunkwise (default) representation of the multi-scale retention mechanism"""
        B, C, _ = value.shape

        # Positional encoding of the current step
        if self.memory_config.timestep_positional_encoding:
            key, query, value = self.pe(key, query, value, step_count)

        q_proj = query @ self.w_q
        k_proj = key @ self.w_k
        v_proj = value @ self.w_v
        k_proj *= self.scaling

        # (B, num_chunks, num_heads, chunk_size, head_size)
        q_proj, k_proj, v_proj = reshape_qkv(q_proj, k_proj, v_proj, self.n_head, num_chunks)
        k_proj = k_proj.transpose(0, 1, 2, -1, -2)

        dones = reshape_dones(dones, num_chunks)

        # we use the timestep ones for computing normalised decay matrix
        _decay_matrix, _xi, _chunk_decay, _delta, _decay_matrix_timesteps, _ = get_decay_matrices(
            dones, self.decay_kappas, self.n_agents, self.masked
        )

        _scale = jnp.sqrt(_decay_matrix_timesteps.sum(axis=-1, keepdims=True))
        _scale = jnp.repeat(_scale, self.n_agents, axis=-2)
        _normalised_decay_matrix = _decay_matrix / _scale

        qk_mat = q_proj @ k_proj
        qk_mat = qk_mat * _normalised_decay_matrix

        # Compute inner scale on timestep level
        qk_mat_timesteps = qk_mat[:, :, :, :: self.n_agents, :: self.n_agents]
        inner_scale = jnp.clip(jnp.abs(qk_mat_timesteps).sum(axis=-1, keepdims=True), min=1.0)
        inner_scale = jnp.repeat(inner_scale, self.n_agents, axis=-2)
        qk_mat = qk_mat / inner_scale
        inner_output = qk_mat @ v_proj

        value_decay_scale_factor = _decay_matrix_timesteps[:, :, :, -1].sum(axis=-1, keepdims=True)
        value_inner_decay = (_decay_matrix[:, :, :, -1] / value_decay_scale_factor)[
            ..., jnp.newaxis
        ]

        _xi_scale_factor = _decay_matrix_timesteps.sum(axis=-1, keepdims=True)
        _xi_scale_factor = jnp.repeat(_xi_scale_factor, self.n_agents, axis=-2)
        _xi = _xi / (_scale / _xi_scale_factor)

        kv = jax.lax.cond(
            inference,
            lambda: k_proj @ v_proj,
            lambda: k_proj @ (v_proj * value_inner_decay),
        )

        kv_recurrent = []
        cross_scale = []

        for chunk in range(num_chunks):
            kv_recurrent.append(hstate / kv_scale)
            cross_scale.append(kv_scale)
            hstate = jax.lax.cond(
                inference,
                lambda hstate=hstate, chunk=chunk: hstate + kv[:, chunk],
                lambda hstate=hstate, chunk=chunk: hstate
                * _chunk_decay[chunk][jnp.newaxis, :, jnp.newaxis, jnp.newaxis]
                * _delta[:, chunk]
                + kv[:, chunk],
            )
            kv_scale = jnp.clip(
                jnp.abs(hstate).sum(axis=-2, keepdims=True).max(axis=-1, keepdims=True), min=1.0
            )

        kv_recurrent = jnp.stack(kv_recurrent, axis=1)
        cross_scale = jnp.stack(cross_scale, axis=1)

        all_scale = jnp.maximum(inner_scale, cross_scale)
        align_inner_scale = all_scale / inner_scale
        align_cross_scale = all_scale / cross_scale

        cross_output = jax.lax.cond(
            inference,
            lambda: q_proj @ kv_recurrent,
            lambda: (q_proj * _xi) @ kv_recurrent,
        )

        ret_output = cross_output / align_cross_scale + inner_output / align_inner_scale

        # Join chunks
        ret_output = rearrange(ret_output, "B nC nh Cs hs -> B nh (nC Cs) hs", nh=self.n_head)

        # Joint heads again
        ret_output = rearrange(ret_output, "B nh C hs -> B C (nh hs)", nh=self.n_head)
        ret_output = self.group_norm(ret_output.reshape(-1, self.head_size)).reshape(
            (B, C, self.embed_dim)
        )

        x = key
        output = (jax.nn.swish(x @ self.w_g) * ret_output) @ self.w_o
        return output, hstate, kv_scale

    def recurrent(
        self,
        key_n: Array,
        query_n: Array,
        value_n: Array,
        hstate: Array,
        step_count: Array,
        kv_scale: Array,
    ) -> Tuple[Array, Array]:
        """Recurrent representation of the multi-scale retention mechanism"""

        B, S, _ = value_n.shape

        # Positional encoding of the current step if enabled
        if self.memory_config.timestep_positional_encoding:
            key_n, query_n, value_n = self.pe(key_n, query_n, value_n, step_count)

        q_proj = query_n @ self.w_q
        k_proj = key_n @ self.w_k
        v_proj = value_n @ self.w_v
        k_proj *= self.scaling

        # Reshape exactly like retnet code
        q_proj = rearrange(q_proj, "B S (nh hs) -> B nh S hs", nh=self.n_head, S=S)
        k_proj = rearrange(k_proj, "B S (nh hs) -> B nh S hs", nh=self.n_head, S=S)
        v_proj = rearrange(v_proj, "B S (nh hs) -> B nh S hs", nh=self.n_head, S=S)

        k_proj = k_proj.transpose(0, 1, 3, 2)

        # TODO: Using mat muls here. This is not exactly like the retnet code.
        # But it was hard to get the chunkwise encoder passes to match up between inference
        # and training. So doing this for now since the tests pass.

        kv = k_proj @ v_proj
        # kv = k_proj * v_proj
        updated_hstate = hstate + kv / jnp.sqrt(kv_scale)
        ret_output = q_proj @ updated_hstate
        # ret_output = (q_proj * updated_hstate).sum(axis=-1)

        # Add back dummy sequence dimension
        # ret_output = ret_output[:, :, jnp.newaxis, :]

        # Rejoin heads
        ret_output = rearrange(ret_output, "B nh S hs -> B S (nh hs)", nh=self.n_head)

        ret_output = self.group_norm(ret_output.reshape(-1, self.head_size)).reshape(
            (B, S, self.embed_dim)
        )

        x = key_n
        output = (jax.nn.swish(x @ self.w_g) * ret_output) @ self.w_o
        return output, updated_hstate
