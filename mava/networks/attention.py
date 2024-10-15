import chex
import jax.numpy as jnp
from flax import linen as nn
from flax.linen.initializers import orthogonal


class SelfAttention(nn.Module):
    n_embd: int
    n_head: int
    n_agent: int
    masked: bool = False

    def setup(self) -> None:
        # flax defualt is to only be defined using features out.
        # bias init is zero in MAT.
        assert self.n_embd % self.n_head == 0
        self.key = nn.Dense(self.n_embd, kernel_init=orthogonal(0.01))
        self.query = nn.Dense(self.n_embd, kernel_init=orthogonal(0.01))
        self.value = nn.Dense(self.n_embd, kernel_init=orthogonal(0.01))

        # output projection
        self.proj = nn.Dense(self.n_embd, kernel_init=orthogonal(0.01))

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.mask = jnp.tril(jnp.ones((self.n_agent + 1, self.n_agent + 1))).reshape(
            1, 1, self.n_agent + 1, self.n_agent + 1
        )

    def __call__(
        self, key: chex.Array, value: chex.Array, query: chex.Array
    ) -> chex.Array:
        # Shape names:
        # B: batch size
        # L: sequence length
        # D: embedding dimension
        # H: number of heads
        # hs: head size
        # nh: number of heads

        batch, seq_len, embed_dim = key.shape

        # calculate query, key, values for all heads in batch and move
        # head forward to be the batch dim
        # [B, L, D] -> [B, L, H, D//H]
        k = (
            self.key(key)
            .reshape(batch, seq_len, self.n_head, embed_dim // self.n_head)
            .transpose((0, 2, 1, 3))
        )
        q = (
            self.query(query)
            .reshape(batch, seq_len, self.n_head, embed_dim // self.n_head)
            .transpose((0, 2, 1, 3))
        )
        v = (
            self.value(value)
            .reshape(batch, seq_len, self.n_head, embed_dim // self.n_head)
            .transpose((0, 2, 1, 3))
        )

        # causal attention: (B, nh, L, hs) x (B, nh, hs, L) -> (B, nh, L, L)
        att = jnp.matmul(q, k.transpose((0, 1, 3, 2))) * (1.0 / jnp.sqrt(k.shape[-1]))

        # mask out attention for all agents
        if self.masked:
            att = jnp.where(
                self.mask[:, :, :seq_len, :seq_len] == 0,
                jnp.finfo(jnp.float32).min,
                att,
            )

        att = nn.softmax(att, axis=-1)

        y = jnp.matmul(att, v)  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        # re-assemble all head outputs side by side
        y = y.transpose((0, 2, 1, 3))
        y = y.reshape(batch, seq_len, embed_dim)

        # output projection
        y = self.proj(y)
        return y  # (B, L, D)