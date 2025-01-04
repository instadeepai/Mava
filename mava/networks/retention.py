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

# General shapes legend:
# B: batch size
# N: number of agents
# S: sequence length
# C: chunk size - T * N in a chunk
# T: number of timesteps


class SimpleRetention(nn.Module):
    """Simple retention mechanism for Sable.

    Note:
        This retention mechanism implementation is based on the following code:
        https://github.com/Jamie-Stirling/RetNet/blob/main/src/retention.py
    """

    embed_dim: int
    head_size: int
    n_agents: int
    masked: bool
    decay_kappa: float  # this is gamma in the original retention implementation
    memory_config: DictConfig

    def setup(self) -> None:
        # Initialise the weights
        self.w_q = self.param(
            "w_q",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.head_size),
        )
        self.w_k = self.param(
            "w_k",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.head_size),
        )
        self.w_v = self.param(
            "w_v",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.head_size),
        )

    def __call__(
        self, key: Array, query: Array, value: Array, hstate: Array, dones: Array
    ) -> Tuple[Array, Array]:
        """Chunkwise (default) representation of the retention mechanism."""
        B, C, _ = value.shape

        # Apply projection to q_proj, k_proj, v_proj
        q_proj = query @ self.w_q
        k_proj = key @ self.w_k
        v_proj = value @ self.w_v
        k_proj = k_proj.transpose(0, -1, -2)

        # Compute next hidden state
        if self.memory_config.type == "ff_sable":
            # No decay matrix or xi for FF Sable since we don't have temporal dependencies.
            decay_matrix = jnp.ones((B, C, C))
            decay_matrix = self._causal_mask(decay_matrix)
            xi = jnp.ones((B, C, 1))
            next_hstate = (k_proj @ v_proj) + hstate
        else:
            decay_matrix = self.get_decay_matrix(dones)
            xi = self.get_xi(dones)
            chunk_decay = self.decay_kappa ** (C // self.n_agents)
            delta = ~jnp.any(dones[:, :: self.n_agents], axis=1)[:, jnp.newaxis, jnp.newaxis]
            next_hstate = (
                k_proj @ (v_proj * decay_matrix[:, -1].reshape((B, C, 1)))
            ) + hstate * chunk_decay * delta

        # Compute the inner chunk and cross chunk
        cross_chunk = (q_proj @ hstate) * xi
        inner_chunk = ((q_proj @ k_proj) * decay_matrix) @ v_proj

        # Compute the final retention
        ret = inner_chunk + cross_chunk
        return ret, next_hstate

    def recurrent(
        self, key_n: Array, query_n: Array, value_n: Array, hstate: Array
    ) -> Tuple[Array, Array]:
        """Recurrent representation of the retention mechanism."""
        # Apply projection to q_proj, k_proj, v_proj
        q_proj = query_n @ self.w_q
        k_proj = key_n @ self.w_k
        v_proj = value_n @ self.w_v

        # Apply the retention mechanism and update the hidden state
        updated_hstate = hstate + (k_proj.transpose(0, -1, -2) @ v_proj)
        ret = q_proj @ updated_hstate

        return ret, updated_hstate

    def get_decay_matrix(self, dones: Array) -> Array:
        """Get the decay matrix for the full sequence based on the dones and retention type."""
        # Extract done information at the timestep level
        timestep_dones = dones[:, :: self.n_agents]  # B, T

        # B, T, T
        timestep_mask = self._get_decay_matrix_mask_timestep(timestep_dones)
        decay_matrix = self._get_default_decay_matrix(timestep_dones)
        decay_matrix *= timestep_mask

        # B, T, T ->  B, T * N, T * N
        decay_matrix = jnp.repeat(
            jnp.repeat(decay_matrix, self.n_agents, axis=1), self.n_agents, axis=2
        )

        # Apply a causal mask over agents if full self-retention is disabled
        # This converts it from a blocked decay matrix to a causal decay matrix
        decay_matrix = self._causal_mask(decay_matrix)

        return decay_matrix

    def _causal_mask(self, matrix: Array) -> Array:
        """Applies a causal mask to the input matrix if `masked` is True."""
        if self.masked:
            mask_agents = jnp.tril(jnp.ones((matrix.shape[1], matrix.shape[1])))
            matrix = mask_agents[None, :, :] * matrix
        return matrix

    def _get_decay_matrix_mask_timestep(self, ts_dones: Array) -> Array:
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

    def _get_default_decay_matrix(self, dones: Array) -> Array:
        """Compute the decay matrix without taking into account the timestep-based masking."""
        # Get the shape of the input: batch size and number of timesteps
        B, T = dones.shape

        # Create the n and m matrices
        n = jnp.arange(T)[:, jnp.newaxis, ...]
        m = jnp.arange(T)[jnp.newaxis, ...]

        # Decay based on difference in timestep indices.
        decay_matrix = (self.decay_kappa ** (n - m)) * (n >= m)
        # Replace NaN values with 0
        decay_matrix = jnp.nan_to_num(decay_matrix)

        # Adjust for batch size
        decay_matrix = jnp.broadcast_to(decay_matrix, (B, T, T))

        return decay_matrix

    def get_xi(self, dones: Array) -> Array:
        """Computes a decaying matrix 'xi', which decays over time until the first done signal."""
        # Get done status for each timestep by slicing out the agent dimension
        timestep_dones = dones[:, :: self.n_agents]
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
            xi_i = (self.decay_kappa ** (i + 1)) * before_first_done
            xi = xi.at[:, i, :].set(xi_i)

        # Repeat the decay matrix 'xi' for all agents
        xi = jnp.repeat(xi, self.n_agents, axis=1)

        return xi


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
        self.group_norm = nn.GroupNorm(num_groups=self.n_head)

        # Initialise the retention mechanisms
        self.retention_heads = [
            SimpleRetention(
                self.embed_dim,
                self.head_size,
                self.n_agents,
                self.masked,
                decay_kappa,
                self.memory_config,
            )
            for decay_kappa in self.decay_kappas
        ]

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
    ) -> Tuple[Array, Array]:
        """Chunkwise (default) representation of the multi-scale retention mechanism"""
        B, C, _ = value.shape

        # Positional encoding of the current step
        if self.memory_config.timestep_positional_encoding:
            key, query, value = self.pe(key, query, value, step_count)

        ret_output = jnp.zeros((B, C, self.embed_dim), dtype=value.dtype)
        for head in range(self.n_head):
            y, new_hs = self.retention_heads[head](key, query, value, hstate[:, head], dones)
            ret_output = ret_output.at[
                :, :, self.head_size * head : self.head_size * (head + 1)
            ].set(y)
            hstate = hstate.at[:, head, :, :].set(new_hs)

        ret_output = self.group_norm(ret_output.reshape(-1, self.head_size)).reshape(
            ret_output.shape
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

        ret_output = jnp.zeros((B, S, self.embed_dim), dtype=value_n.dtype)
        for head in range(self.n_head):
            y, new_hs = self.retention_heads[head].recurrent(
                key_n, query_n, value_n, hstate[:, head]
            )
            ret_output = ret_output.at[
                :, :, self.head_size * head : self.head_size * (head + 1)
            ].set(y)
            hstate = hstate.at[:, head, :, :].set(new_hs)

        ret_output = self.group_norm(ret_output.reshape(-1, self.head_size)).reshape(
            ret_output.shape
        )

        x = key_n
        output = (jax.nn.swish(x @ self.w_g) * ret_output) @ self.w_o
        return output, hstate
