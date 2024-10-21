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
from flax.linen.initializers import orthogonal

from mava.networks.attention import SelfAttention
from mava.networks.utils.mat.decode import (
    continuous_autoregressive_act,
    continuous_parallel_act,
    discrete_autoregressive_act,
    discrete_parallel_act,
)


class SwiGLU(nn.Module):
    ffn_dim: int
    embed_dim: int

    def setup(self) -> None:
        self.W_1 = self.param("W_1", nn.initializers.zeros, (self.embed_dim, self.ffn_dim))
        self.W_G = self.param("W_G", nn.initializers.zeros, (self.embed_dim, self.ffn_dim))
        self.W_2 = self.param("W_2", nn.initializers.zeros, (self.ffn_dim, self.embed_dim))

    def __call__(self, x: chex.Array) -> chex.Array:
        return (jax.nn.swish(x @ self.W_G) * (x @ self.W_1)) @ self.W_2


class EncodeBlock(nn.Module):
    n_embd: int
    n_head: int
    n_agent: int
    masked: bool = False
    use_swiglu: bool = False
    use_rmsnorm: bool = False

    def setup(self) -> None:
        ln = nn.RMSNorm if self.use_rmsnorm else nn.LayerNorm
        self.ln1 = ln()
        self.ln2 = ln()

        self.attn = SelfAttention(self.n_embd, self.n_head, self.n_agent, self.masked)

        if self.use_swiglu:
            self.mlp = SwiGLU(self.n_embd, self.n_embd)
        else:
            self.mlp = nn.Sequential(
                [
                    nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                    nn.gelu,
                    nn.Dense(self.n_embd, kernel_init=orthogonal(0.01)),
                ],
            )

    def __call__(self, x: chex.Array) -> chex.Array:
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x


class Encoder(nn.Module):
    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_head: int
    n_agent: int
    use_swiglu: bool = False
    use_rmsnorm: bool = False

    def setup(self) -> None:
        ln = nn.RMSNorm if self.use_rmsnorm else nn.LayerNorm

        self.obs_encoder = nn.Sequential(
            [ln(), nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu],
        )
        self.ln = ln()
        self.blocks = nn.Sequential(
            [
                EncodeBlock(
                    self.n_embd,
                    self.n_head,
                    self.n_agent,
                    use_swiglu=self.use_swiglu,
                    use_rmsnorm=self.use_swiglu,
                )
                for _ in range(self.n_block)
            ]
        )
        self.head = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                ln(),
                nn.Dense(1, kernel_init=orthogonal(0.01)),
            ],
        )

    def __call__(self, obs: chex.Array) -> Tuple[chex.Array, chex.Array]:
        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings

        rep = self.blocks(self.ln(x))
        v_loc = self.head(rep)

        return jnp.squeeze(v_loc, axis=-1), rep


class DecodeBlock(nn.Module):
    n_embd: int
    n_head: int
    n_agent: int
    masked: bool = True
    use_swiglu: bool = False
    use_rmsnorm: bool = False

    def setup(self) -> None:
        ln = nn.RMSNorm if self.use_rmsnorm else nn.LayerNorm
        self.ln1 = ln()
        self.ln2 = ln()
        self.ln3 = ln()

        self.attn1 = SelfAttention(self.n_embd, self.n_head, self.n_agent, self.masked)
        self.attn2 = SelfAttention(self.n_embd, self.n_head, self.n_agent, self.masked)

        if self.use_swiglu:
            self.mlp = SwiGLU(self.n_embd, self.n_embd)
        else:
            self.mlp = nn.Sequential(
                [
                    nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                    nn.gelu,
                    nn.Dense(self.n_embd, kernel_init=orthogonal(0.01)),
                ],
            )

    def __call__(self, x: chex.Array, rep_enc: chex.Array) -> chex.Array:
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x


class Decoder(nn.Module):
    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_head: int
    n_agent: int
    action_space_type: str = "discrete"
    use_swiglu: bool = False
    use_rmsnorm: bool = False

    def setup(self) -> None:
        ln = nn.RMSNorm if self.use_rmsnorm else nn.LayerNorm

        if self.action_space_type == "discrete":
            self.action_encoder = nn.Sequential(
                [
                    nn.Dense(self.n_embd, use_bias=False, kernel_init=orthogonal(jnp.sqrt(2))),
                    nn.gelu,
                ],
            )
        else:
            self.action_encoder = nn.Sequential(
                [nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu],
            )
            self.log_std = self.param("log_std", nn.initializers.zeros, (self.action_dim,))

        # Always initialize log_std but set to None for discrete action spaces
        # This ensures the attribute exists but signals it should not be used.
        if self.action_space_type == "discrete":
            self.log_std = None

        self.obs_encoder = nn.Sequential(
            [ln(), nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))), nn.gelu],
        )
        self.ln = ln()
        self.blocks = [
            DecodeBlock(
                self.n_embd,
                self.n_head,
                self.n_agent,
                use_swiglu=self.use_swiglu,
                use_rmsnorm=self.use_swiglu,
                name=f"cross_attention_block_{block_id}",
            )
            for block_id in range(self.n_block)
        ]
        self.head = nn.Sequential(
            [
                nn.Dense(self.n_embd, kernel_init=orthogonal(jnp.sqrt(2))),
                nn.gelu,
                ln(),
                nn.Dense(self.action_dim, kernel_init=orthogonal(0.01)),
            ],
        )

    def __call__(self, action: chex.Array, obs_rep: chex.Array) -> chex.Array:
        action_embeddings = self.action_encoder(action)
        x = self.ln(action_embeddings)

        # Need to loop here because the input and output of the blocks are different.
        # Blocks take an action embedding and observation encoding as input but only give the cross
        # attention output as output.
        for block in self.blocks:
            x = block(x, obs_rep)
        logit = self.head(x)

        return logit


class MultiAgentTransformer(nn.Module):
    obs_dim: int
    action_dim: int
    n_block: int
    n_embd: int
    n_head: int
    n_agent: int
    action_space_type: str = "discrete"
    use_swiglu: bool = False
    use_rmsnorm: bool = False

    # General shapes legend:
    # B: batch size
    # N: number of agents
    # O: observation dimension
    # A: action dimension
    # E: model embedding dimension

    def setup(self) -> None:
        if self.action_space_type not in ["discrete", "continuous"]:
            raise ValueError(f"Invalid action space type: {self.action_space_type}")

        self.encoder = Encoder(
            self.obs_dim,
            self.action_dim,
            self.n_block,
            self.n_embd,
            self.n_head,
            self.n_agent,
            use_swiglu=self.use_swiglu,
            use_rmsnorm=self.use_rmsnorm,
        )
        self.decoder = Decoder(
            self.obs_dim,
            self.action_dim,
            self.n_block,
            self.n_embd,
            self.n_head,
            self.n_agent,
            self.action_space_type,
            use_swiglu=self.use_swiglu,
            use_rmsnorm=self.use_rmsnorm,
        )

        if self.action_space_type == "discrete":
            self.act_function = discrete_autoregressive_act
            self.train_function = discrete_parallel_act
        elif self.action_space_type == "continuous":
            self.act_function = continuous_autoregressive_act
            self.train_function = continuous_parallel_act
        else:
            raise ValueError(f"Invalid action space type: {self.action_space_type}")

    def __call__(
        self,
        obs: chex.Array,  # (B, N, O)
        action: chex.Array,  # (B, N, A)
        legal_actions: chex.Array,  # (B, N, A)
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        v_loc, obs_rep = self.encoder(obs)

        action_log, entropy = self.train_function(
            decoder=self.decoder,
            obs_rep=obs_rep,
            action=action,
            action_dim=self.action_dim,
            legal_actions=legal_actions,
            key=key,
        )

        return action_log, v_loc, entropy

    def get_actions(
        self,
        obs: chex.Array,  # (B, N, O)
        legal_actions: chex.Array,  # (B, N, A)
        key: chex.PRNGKey,
    ) -> Tuple[chex.Array, chex.Array, chex.Array]:
        v_loc, obs_rep = self.encoder(obs)
        output_action, output_action_log = self.act_function(
            decoder=self.decoder,
            obs_rep=obs_rep,
            action_dim=self.action_dim,
            legal_actions=legal_actions,
            key=key,
        )
        return output_action, output_action_log, v_loc
