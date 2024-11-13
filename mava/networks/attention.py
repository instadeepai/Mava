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

import chex
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal

# TODO: Use einops for all the reshapes and matrix multiplications


class SelfAttention(nn.Module):
    embed_dim: int
    n_head: int
    n_agent: int
    masked: bool = False

    def setup(self) -> None:
        assert self.embed_dim % self.n_head == 0
        self.key = nn.Dense(self.embed_dim, kernel_init=orthogonal(0.01))
        self.query = nn.Dense(self.embed_dim, kernel_init=orthogonal(0.01))
        self.value = nn.Dense(self.embed_dim, kernel_init=orthogonal(0.01))

        # output projection
        self.proj = nn.Dense(self.embed_dim, kernel_init=orthogonal(0.01))

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.mask = jnp.tril(jnp.ones((self.n_agent + 1, self.n_agent + 1)))
        self.mask = self.mask[jnp.newaxis, jnp.newaxis]

    def __call__(self, key: chex.Array, value: chex.Array, query: chex.Array) -> chex.Array:
        # Shape names:
        # B: batch size
        # S: sequence length
        # E: embedding dimension
        # hs: head size
        # nh: number of heads

        B, S, D = key.shape

        # calculate query, key, values for all heads in batch and move
        # head forward to be the batch dim
        # (B, S, E) -> (B, nh, S, hs)
        k = self.key(key).reshape(B, S, self.n_head, D // self.n_head).transpose((0, 2, 1, 3))
        q = self.query(query).reshape(B, S, self.n_head, D // self.n_head).transpose((0, 2, 1, 3))
        v = self.value(value).reshape(B, S, self.n_head, D // self.n_head).transpose((0, 2, 1, 3))

        # causal attention: (B, nh, S, hs) x (B, nh, hs, S) -> (B, nh, S, S)
        att = jnp.matmul(q, k.transpose((0, 1, 3, 2))) * (1.0 / jnp.sqrt(k.shape[-1]))

        # mask out attention for all agents
        if self.masked:
            att = jnp.where(
                self.mask[:, :, :S, :S] == 0,
                jnp.finfo(jnp.float32).min,
                att,
            )

        att = nn.softmax(att, axis=-1)

        y = jnp.matmul(att, v)  # (B, nh, S, S) x (B, nh, S, hs) -> (B, nh, S, hs)
        # re-assemble all head outputs side by side
        y = y.transpose((0, 2, 1, 3))
        y = y.reshape(B, S, D)

        return self.proj(y)  # (B, S, D)
