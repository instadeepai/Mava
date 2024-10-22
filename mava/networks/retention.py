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

from mava.utils.sable_utils import PositionalEncoding


class SimpleRetention(nn.Module):
    """Simple retention mechanism for Sable."""

    embed_dim: int
    head_size: int
    n_agents: int
    full_self_retention: bool
    decay_kappa: float
    net_config: DictConfig

    def setup(self) -> None:
        # Initialize the weights
        self.W_Q = self.param(
            "W_Q",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.head_size),
        )
        self.W_K = self.param(
            "W_K",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.head_size),
        )
        self.W_V = self.param(
            "W_V",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.head_size),
        )

    def __call__(
        self, key: Array, query: Array, value: Array, hstate: Array, dones: Array
    ) -> Tuple[Array, Array]:
        """Chunkwise (default) representation of the retention mechanism."""
        batch, chunk_size, _ = value.shape

        # Apply projection to q_proj, k_proj, v_proj
        q_proj = query @ self.W_Q
        k_proj = key @ self.W_K
        v_proj = value @ self.W_V
        k_proj = k_proj.transpose(0, -1, -2)

        # Compute next hidden state
        decay_matrix = self.get_decay_matrix(dones)
        xi = self.get_xi(dones)
        if self.net_config.type == "ff_sable":
            next_hstate = (k_proj @ v_proj) + hstate
            decay_matrix = jnp.ones_like(decay_matrix)
            xi = jnp.ones_like(xi)
        else:
            chunk_decay = self.decay_kappa ** (chunk_size // self.n_agents)
            delta = ~jnp.any(dones[:, :: self.n_agents], axis=1)[:, jnp.newaxis, jnp.newaxis]
            next_hstate = (
                k_proj @ (v_proj * decay_matrix[:, -1].reshape((batch, chunk_size, 1)))
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
        q_proj = query_n @ self.W_Q
        k_proj = key_n @ self.W_K
        v_proj = value_n @ self.W_V

        # Apply the retention mechanism and update the hidden state
        updated_hstate = hstate + (k_proj.transpose(0, -1, -2) @ v_proj)
        ret = q_proj @ updated_hstate

        return ret, updated_hstate

    def get_decay_matrix(self, dones: Array) -> Array:
        """Get the decay matrix for the full sequence based on the dones and retention type."""
        # Extract done information at the timestep level
        timestep_dones = dones[:, :: self.n_agents]

        # Compute the decay matrix and apply timestep-based masking
        decay_matrix = self._get_default_decay_matrix(
            timestep_dones
        ) * self._get_decay_matrix_mask_timestep(timestep_dones)

        # Repeat decay matrix across agents
        decay_matrix = jnp.repeat(
            jnp.repeat(decay_matrix, self.n_agents, axis=1), self.n_agents, axis=2
        )

        # Apply a causal mask over agents if full self-retention is disabled
        if not self.full_self_retention:
            mask_agents = jnp.tril(jnp.ones((decay_matrix.shape[1], decay_matrix.shape[1])))
            decay_matrix = mask_agents[None, :, :] * decay_matrix

        return decay_matrix

    def _get_decay_matrix_mask_timestep(self, ts_dones: Array) -> Array:
        """Generates a mask over the timesteps based on the done status of agents."""
        # Get the shape of the input: batch size and number of timesteps
        batch, num_ts = ts_dones.shape

        # Initialize the mask
        timestep_mask = jnp.zeros((batch, num_ts, num_ts), dtype=bool)
        all_false = jnp.zeros((batch, num_ts, num_ts), dtype=bool)

        # Iterate over the timesteps and apply the mask
        for i in range(num_ts):
            done_this_step = ts_dones[:, i, jnp.newaxis, jnp.newaxis]
            # Block positions below the current timestep
            ts_done_xs = all_false.at[:, i:, :].set(done_this_step)
            # Block positions before the current timestep
            ts_done_ys = all_false.at[:, :, :i].set(done_this_step)

            # Combine the x and y masks to get the mask for the current timestep.
            timestep_mask |= ts_done_xs & ts_done_ys

        return ~timestep_mask

    def _get_default_decay_matrix(self, dones: Array) -> Array:
        """Compute the decay matrix without taking into account the timestep-based masking."""
        # Get the shape of the input: batch size and number of timesteps
        batch, num_ts = dones.shape

        # Create the n and m matrices
        n = jnp.arange(num_ts)[:, jnp.newaxis, ...]
        m = jnp.arange(num_ts)[jnp.newaxis, ...]

        # Decay based on difference in timestep indices.
        decay_matrix = (self.decay_kappa ** (n - m)) * (n >= m)
        # Replace NaN values with 0
        decay_matrix = jnp.nan_to_num(decay_matrix)

        # Adjust for batch size
        decay_matrix = jnp.broadcast_to(decay_matrix, (batch, num_ts, num_ts))

        return decay_matrix

    def get_xi(self, dones: Array) -> Array:
        """Computes a decaying matrix 'xi', which decays over time until the first done signal."""
        # Get done status for each timestep by slicing out the agent dimension
        timestep_dones = dones[:, :: self.n_agents]
        batch, num_ts = timestep_dones.shape

        # Compute the first done step for each sequence,
        # or set it to sequence length if no dones exist
        first_dones = jnp.where(
            ~jnp.any(timestep_dones, axis=1, keepdims=True),
            jnp.full((batch, 1), num_ts),
            jnp.argmax(timestep_dones, axis=1, keepdims=True),
        )

        # Initialize the decaying matrix 'xi'
        xi = jnp.zeros((batch, num_ts, 1))
        # Fill 'xi' with decaying values up until the first done step
        for i in range(num_ts):
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
    net_config: DictConfig
    full_self_retention: bool = False
    decay_scaling_factor: float = 1.0

    def setup(self) -> None:
        assert self.embed_dim % self.n_head == 0, "embed_dim must be divisible by n_head"
        # Head size
        self.head_size = self.embed_dim // self.n_head

        # Decay kappa for each head
        self.decay_kappas = 1 - jnp.exp(
            jnp.linspace(jnp.log(1 / 32), jnp.log(1 / 512), self.n_head)
        )
        self.decay_kappas = self.decay_kappas * self.decay_scaling_factor

        # Initialize the weights and group norm
        self.W_G = self.param(
            "W_G",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.embed_dim, self.head_size),
        )
        self.W_O = self.param(
            "W_O",
            nn.initializers.normal(stddev=1 / self.embed_dim),
            (self.head_size, self.embed_dim),
        )
        self.group_norm = nn.GroupNorm(num_groups=self.n_head)

        # Initialize the retention mechanisms
        self.retentions = [
            SimpleRetention(
                self.embed_dim,
                self.head_size,
                self.n_agents,
                self.full_self_retention,
                decay_kappa,
                self.net_config,
            )
            for decay_kappa in self.decay_kappas
        ]

        # Create an instance of the positional encoding
        self.pe = PositionalEncoding(self.net_config, self.embed_dim)

    def __call__(
        self,
        key: Array,
        query: Array,
        value: Array,
        hstate: Array,
        dones: Array,
        timestep_id: Array,
    ) -> Tuple[Array, Array]:
        """Chunkwise (default) representation of the multi-scale retention mechanism"""
        batch, chunk_size, _ = value.shape

        # Set positional encoding
        key, query, value = self.pe(key, query, value, timestep_id)

        # Per head retention
        ret_output = jnp.zeros((batch, chunk_size, self.head_size), dtype=value.dtype)
        h_ns = jnp.copy(hstate)
        for head in range(self.n_head):
            y, h_n = self.retentions[head](key, query, value, hstate[:, head], dones)
            ret_output = ret_output.at[:, :, self.head_size : (self.head_size + 1)].set(y)
            h_ns = h_ns.at[:, head, :, :].set(h_n)

        # Gated Multi-scale retention
        # Apply the group norm
        ret_output = self.group_norm(ret_output.reshape(-1, self.head_size)).reshape(
            ret_output.shape
        )

        # Swish gating
        x = key
        output = (jax.nn.swish(x @ self.W_G) * ret_output) @ self.W_O
        return output, h_ns

    def recurrent(
        self, key_n: Array, query_n: Array, value_n: Array, hstate: Array, timestep_id: Array
    ) -> Tuple[Array, Array]:
        """Recurrent representation of the multi-scale retention mechanism"""
        batch, seq, _ = value_n.shape

        # Set positional encoding
        key_n, query_n, value_n = self.pe(key_n, query_n, value_n, timestep_id)

        # Per head retention
        ret_output = jnp.zeros((batch, seq, self.head_size), dtype=value_n.dtype)
        h_ns = jnp.zeros_like(hstate)
        for head in range(self.n_head):
            y, h_n = self.retentions[head].recurrent(key_n, query_n, value_n, hstate[:, head])
            ret_output = ret_output.at[:, :, self.head_size : (self.head_size + 1)].set(y)
            h_ns = h_ns.at[:, head, :, :].set(h_n)

        # Gated Multi-scale retention
        # Apply the group norm
        ret_output = self.group_norm(ret_output.reshape(-1, self.head_size)).reshape(
            ret_output.shape
        )

        # Swish gating
        x = key_n
        output = (jax.nn.swish(x @ self.W_G) * ret_output) @ self.W_O
        return output, h_ns
