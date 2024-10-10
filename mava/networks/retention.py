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
from colorama import Fore, Style


class SimpleRetention(nn.Module):
    """Simple retention mechanism for Sable."""
    n_embd: int
    head_size: int
    n_agents: int
    full_self_retention: bool
    decay_kappa: float

    def setup(self)->None:
        # Initialize the weights
        self.W_Q = self.param(
            "W_Q",
            nn.initializers.normal(stddev=1 / self.n_embd),
            (self.n_embd, self.head_size),
        )
        self.W_K = self.param(
            "W_K",
            nn.initializers.normal(stddev=1 / self.n_embd),
            (self.n_embd, self.head_size),
        )
        self.W_V = self.param(
            "W_V",
            nn.initializers.normal(stddev=1 / self.n_embd),
            (self.n_embd, self.head_size),
        )

    def __call__(self, key: Array, query: Array, value: Array)->Array:
        """Parallel (default) representation of the retention mechanism."""
        # Apply projection to Q, K, V
        Q = query @ self.W_Q
        K = key @ self.W_K
        V = value @ self.W_V

        # Apply the retention mechanism
        ret = Q @ jnp.transpose(K, (0, 2, 1))

        # Apply causal mask if not full self retention
        if not self.full_self_retention:
            D= jnp.tril(jnp.ones((K.shape[1], K.shape[1])))
            ret = (ret * D)

        ret = ret @ V
        return ret

    def recurrent(self, key_n: Array, query_n: Array, value_n: Array, hstate: Array)-> Tuple[Array, Array]:
        """Recurrent representation of the retention mechanism."""
        # Apply projection to Q, K, V
        Q = query_n @ self.W_Q
        K = key_n @ self.W_K
        V = value_n @ self.W_V

        # Apply the retention mechanism and update the hidden state
        updated_hstate = hstate + (K.transpose(0, -1, -2) @ V)
        ret = Q @ updated_hstate

        return ret, updated_hstate

    def chunkwise(self, key: Array, query: Array, value: Array, hstate: Array, dones: Array)-> Tuple[Array, Array]:
        """Chunkwise representation of the retention mechanism."""
        batch, chunk_size, _ = value.shape

        # Apply projection to Q, K, V
        Q = query @ self.W_Q
        K = key @ self.W_K
        V = value @ self.W_V
        K = K.transpose(0, -1, -2)
        
        # Compute next hidden state
        D = self.get_masked_D(dones)
        chunk_decay = self.decay_kappa ** (chunk_size // self.n_agents)
        delta = ~jnp.any(dones[:, :: self.n_agents], axis=1)[
            :, jnp.newaxis, jnp.newaxis
        ]
        next_hstate = (
            K @ (V * D[:, -1].reshape((batch, chunk_size, 1)))
        ) + hstate * chunk_decay * delta

        # Compute the inner chunk and cross chunk
        e = self.get_masked_e(dones)
        cross_chunk = (Q @ hstate) * e
        inner_chunk = ((Q @ K) * D) @ V

        # Compute the final retention
        ret = inner_chunk + cross_chunk
        return ret, next_hstate

    def get_masked_D(self, dones):
        # get dones the timestep as we do full self attention over all agents within a timestep
        timestep_dones = dones[:, :: self.n_agents]
        D = self._get_D(timestep_dones) * self._get_D_mask_timestep(timestep_dones)
        # repeat D matrix to represent the blocks of full self attention - ie repeat per agent
        D = jnp.repeat(jnp.repeat(D, self.n_agents, axis=1), self.n_agents, axis=2)

        if not self.full_self_attn:
            # Causal mask over the agents
            mask_agents = jnp.tril(jnp.ones((D.shape[1], D.shape[1])))
            D = mask_agents[None, :, :] * D

        return D

    def _get_D_mask_timestep(self, ts_dones):
        """Get the D mask over the timestep. The case used for full self attention.

        Args:
            dones: (batch_size, sequence_length) boolean array of dones -
            must contain only the dones per timestep! Sequence length = num_timesteps.
        """
        B, S = ts_dones.shape
        ts_done_mask = jnp.zeros((B, S, S), dtype=bool)
        # create a blank mask used for indexing
        all_false = jnp.zeros((B, S, S), dtype=bool)
        for i in range(S):
            done_this_step = ts_dones[:, i, jnp.newaxis, jnp.newaxis]
            # We always want to mask below and left of the current done.
            # Therefore create a mask for all the xs below and ys left of the current done.
            ts_done_xs = all_false.at[:, i:, :].set(done_this_step)
            ts_done_ys = all_false.at[:, :, :i].set(done_this_step)

            # Combine the x and y masks to get the mask for the current timestep.
            ts_done_mask |= ts_done_xs & ts_done_ys

        return ~ts_done_mask

    def _get_D(self, dones):
        B, S = dones.shape
        n = jnp.arange(S)[:, jnp.newaxis, ...]
        m = jnp.arange(S)[jnp.newaxis, ...]

        # Broadcast self.decay_kappa ** (n - m) with appropriate masking to set values where n < m to 0
        D = (self.decay_kappa ** (n - m)) * (n >= m)
        # this results in some NaN when n is much larger than m
        # fill the NaN with 0
        D = jnp.nan_to_num(D)
        # create a decay mat for each batch as each batch will have different dones
        D = jnp.broadcast_to(D, (B, S, S))

        return D

    def get_masked_e(self, dones):
        # normal e until first done, then zeros
        dones = dones[:, :: self.n_agents]

        B, S = dones.shape

        # If there is no done this sequence then the first done is after the
        # sequence so must set it to chunksize, as argmax would return 0.
        first_dones = jnp.where(
            ~jnp.any(dones, axis=1, keepdims=True),
            jnp.full((B, 1), S),
            jnp.argmax(dones, axis=1, keepdims=True),
        )
        e = jnp.zeros((B, S, 1))
        for i in range(S):
            before_first_done = i < first_dones
            e_i = (self.decay_kappa ** (i + 1)) * before_first_done
            e = e.at[:, i, :].set(e_i)

        e = jnp.repeat(e, self.n_agents, axis=1)

        return e


class MultiScaleRetention(nn.Module):
    """Multi-scale retention mechanism for Sable."""
    n_embd: int
    n_head: int
    n_agents: int
    full_self_retention: bool = False
    decay_scaling_factor: float = 1.0

    def setup(self)->None:
        assert self.n_embd % self.n_head == 0, "n_embd must be divisible by n_head"
        # Head size
        self.head_size = self.n_embd // self.n_head

        # Decay kappa for each head
        self.decay_kappas = 1 - jnp.exp(jnp.linspace(jnp.log(1 / 32), jnp.log(1 / 512), self.n_head))
        self.decay_kappas = self.decay_kappas * self.decay_scaling_factor
        if self.decay_scaling_factor <= 0:
            self.decay_kappas = jnp.ones_like(self.decay_kappas) # No decaying if decay_scaling_factor is 0
            print(
                f"{Fore.RED}{Style.BRIGHT} No decaying will be applied to the sequence because decay_scaling_factor is 0.{Style.RESET_ALL}"
            )

        # Initialize the weights and group norm
        self.W_G = self.param(
            "W_G",
            nn.initializers.normal(stddev=1 / self.n_embd),
            (self.n_embd, self.head_size),
        )
        self.W_O = self.param(
            "W_O",
            nn.initializers.normal(stddev=1 / self.n_embd),
            (self.head_size, self.n_embd),
        )
        self.group_norm = nn.GroupNorm(num_groups=self.heads)

        # Initialize the retention mechanisms
        self.retentions = [
            SimpleRetention(
                self.n_embd,
                self.head_size,
                self.n_agents,
                self.full_self_retention,
                decay_kappa,
            )
            for decay_kappa in self.decay_kappas
        ]

    def __call__(self, key: Array, query: Array, value: Array)->Array:
        """Parallel (default) representation of the multi-scale retention mechanism."""
        batch, seq_len, _ = value.shape

        assert (seq_len == self.n_agents), "Parallel retention expects a sequence equal to the number of agents."

        # Per head retention
        Y = jnp.zeros((batch, seq_len, self.head_size), dtype=value.dtype)
        for head in range(self.n_head):
            Y = Y.at[:, :, head * self.v_dim_head_size : (head + 1)].set(
                self.retentions[head](key, query, value)
            )

        # Gated Multi-scale retention
        # Apply the group norm
        Y = self.group_norm(Y.reshape(-1, self.head_size)).reshape(Y.shape)

        # Swish gating
        X = key
        output= (jax.nn.swish(X @ self.W_G) * Y) @ self.W_O
        return output

    def recurrent(self, key_n: Array, query_n: Array, value_n: Array, hstate: Array)-> Tuple[Array, Array]:
        """ Recurrent representation of the multi-scale retention mechanism"""
        batch = value_n.shape[0]

        # Per head retention
        Y = jnp.zeros((batch, 1, self.head_size), dtype=value_n.dtype)
        h_ns = jnp.zeros_like(hstate)
        for head in range(self.n_head):
            y, h_n = self.retentions[head].recurrent(key_n, query_n, value_n, hstate[:, head])
            Y = Y.at[:, :, head : (head + 1)].set(y)
            h_ns = h_ns.at[:, head, :, :].set(h_n)

        # Gated Multi-scale retention
        # Apply the group norm
        Y = self.group_norm(Y.reshape(-1, self.head_size)).reshape(Y.shape)
        
        # Swish gating
        X = key_n
        output = (jax.nn.swish(X @ self.W_G) * Y) @ self.W_O
        return output, h_ns

    def chunkwise(self, key: Array, query: Array, value: Array, hstate: Array, dones: Array)-> Tuple[Array, Array]:
        """ Chunkwise representation of the multi-scale retention mechanism"""
        batch, chunk_size, _ = value.shape

        # Per head retention
        Y = jnp.zeros((batch, chunk_size, self.head_size), dtype=value.dtype)
        h_ns = jnp.copy(hstate)
        for head in range(self.n_head):
            y, h_n = self.retentions[i].chunkwise(
                key, query, value, hstate[:, head], dones
            )
            Y = Y.at[:, :, head : (head + 1)].set(y)
            h_ns = h_ns.at[:, head, :, :].set(h_n)

        # Gated Multi-scale retention
        # Apply the group norm
        Y = self.group_norm(Y.reshape(-1, self.head_size)).reshape(Y.shape)
        
        # Swish gating
        X = key
        output = (jax.nn.swish(X @ self.W_G) * Y) @ self.W_O
        return output, h_ns
