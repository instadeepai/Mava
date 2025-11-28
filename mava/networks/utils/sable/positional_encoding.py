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
import jax
import jax.numpy as jnp
from flax import linen as nn


class PositionalEncoding(nn.Module):
    """Positional Encoding for Sable. Encodes position information into sequences"""

    d_model: int

    def setup(self) -> None:
        # Set maximum sequence length for positional encoding
        self.max_size = 10_000
        # Precompute the scaling factor for even indices (used in sine and cosine functions)
        self.div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model)
        )[jnp.newaxis]

    def __call__(
        self, key: chex.Array, query: chex.Array, value: chex.Array, position: chex.Array
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Computes positional encoding for a given sequence of positions."""
        pe = jax.vmap(self._get_pos_encoding)(position)

        # Add positional encoding to the input tensors
        key += pe
        query += pe
        value += pe

        return key, query, value

    def _get_pos_encoding(self, position: chex.Array) -> chex.Array:
        """Computes positional encoding for a given the index of the token."""
        seq_len = position.shape[0]

        # Calculate positional encoding using sine for even indices and cosine for odd indices.
        x = position[:, jnp.newaxis] * self.div_term
        pe = jnp.zeros((seq_len, self.d_model))
        pe = pe.at[:, 0::2].set(jnp.sin(x))
        pe = pe.at[:, 1::2].set(jnp.cos(x))

        return pe


class XPOSPositionalEncoding(nn.Module):
    """XPOS rotary positional encoding.

    Assumes inputs are already-projected [batch, seq_len, d_model] matrices.
    Applies XPOS (RoPE + magnitude scaling) to key and query; value is
    passed through unchanged.
    """

    d_model: int
    base: float = 10_000.0  # RoPE frequency base
    scale_base: float = 512.0  # XPOS magnitude base (used in log_scale)
    scale_norm: float = 512.0  # position normalisation for stability

    def setup(self) -> None:
        assert self.d_model % 2 == 0, "d_model must be even for rotary/XPOS."

        # Frequencies for RoPE-style angles (one per even/odd pair)
        dim = jnp.arange(0, self.d_model, 2, dtype=jnp.float32)  # [d_model//2]
        self.inv_freq = 1.0 / (self.base ** (dim / self.d_model))  # [d_model//2]

        # Canonical XPOS log-scale (independent of gamma/decay_kappa)
        self.log_scale = jnp.linspace(-1.0, 1.0, self.d_model // 2, dtype=jnp.float32) * jnp.log(
            self.scale_base
        )  # [d_model//2]

    def __call__(
        self,
        query: chex.Array,  # [batch, seq_len, d_model]
        key: chex.Array,  # [batch, seq_len, d_model]
        value: chex.Array,  # [batch, seq_len, d_model]
        position: chex.Array,  # [batch, seq_len] or [seq_len]
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Rotate query/key using XPOS; leave value unchanged."""
        assert key.shape == query.shape == value.shape
        b, t, d = key.shape
        assert d == self.d_model

        # --- Positions ---
        pos = position
        if pos.ndim == 1:
            # Broadcast [seq_len] -> [batch, seq_len]
            pos = jnp.broadcast_to(pos[None, :], (b, t))
        else:
            assert pos.shape == (b, t)
        pos_f = pos.astype(jnp.float32)  # [b, t]

        # --- RoPE angles ---
        # half = self.d_model // 2
        # inv_freq: [half] -> [b, t, half]
        freqs = pos_f[..., None] * self.inv_freq  # [b, t, half]
        cos = jnp.cos(freqs)  # [b, t, half]
        sin = jnp.sin(freqs)  # [b, t, half]

        # --- XPOS magnitude scaling (no gamma coupling) ---
        # Normalise positions to keep scaling gentle
        pos_log_scale = (pos_f[..., None] / self.scale_norm) * self.log_scale  # [b, t, half]
        scale_q = jnp.exp(pos_log_scale)  # [b, t, half]
        scale_k = jnp.exp(-pos_log_scale)  # [b, t, half]

        def split_even_odd(x: chex.Array) -> Tuple[chex.Array, chex.Array]:
            # x: [b, t, d_model]
            return x[..., 0::2], x[..., 1::2]  # [b, t, half] each

        def merge_even_odd(even: chex.Array, odd: chex.Array) -> chex.Array:
            x = jnp.stack([even, odd], axis=-1)  # [b, t, half, 2]
            return x.reshape(b, t, -1)  # [b, t, d_model]

        def rotate(
            even: chex.Array,
            odd: chex.Array,
            cos: chex.Array,
            sin: chex.Array,
        ) -> Tuple[chex.Array, chex.Array]:
            e_rot = even * cos - odd * sin
            o_rot = even * sin + odd * cos
            return e_rot, o_rot

        # --- Split ---
        q_even, q_odd = split_even_odd(query)
        k_even, k_odd = split_even_odd(key)

        # --- XPOS scaling ---
        q_even = q_even * scale_q
        q_odd = q_odd * scale_q
        k_even = k_even * scale_k
        k_odd = k_odd * scale_k

        # --- Rotary phase ---
        q_even_rot, q_odd_rot = rotate(q_even, q_odd, cos, sin)
        k_even_rot, k_odd_rot = rotate(k_even, k_odd, cos, sin)

        # --- Merge back to full d_model ---
        query_rot = merge_even_odd(q_even_rot, q_odd_rot)
        key_rot = merge_even_odd(k_even_rot, k_odd_rot)

        # Value is left unchanged
        return query_rot, key_rot, value
