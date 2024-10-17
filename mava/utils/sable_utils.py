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
import jax
import jax.numpy as jnp
from flax import linen as nn
from omegaconf import DictConfig
from typing import Tuple
from mava.systems.sable.types import HiddenStates


class SwiGLU(nn.Module):
    """SiwGLU module for Sable's Network.

    Implements the SwiGLU feedforward neural network module, which is a variation
    of the standard feedforward layer using the Swish activation function combined
    with a Gated Linear Unit (GLU).
    """

    hidden_dim: int
    input_dim: int

    def setup(self) -> None:
        # Initialize the weights for the SwiGLU layer
        self.W_linear = self.param(
            "W_linear", nn.initializers.zeros, (self.input_dim, self.hidden_dim)
        )
        self.W_gate = self.param("W_gate", nn.initializers.zeros, (self.input_dim, self.hidden_dim))
        self.W_output = self.param(
            "W_output", nn.initializers.zeros, (self.hidden_dim, self.input_dim)
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        """Applies the SwiGLU mechanism to the input tensor `x`."""
        # Apply Swish activation to the gated branch and multiply with the linear branch
        gated_output = jax.nn.swish(x @ self.W_gate) * (x @ self.W_linear)
        # Transform the result back to the input dimension
        return gated_output @ self.W_output


def concat_time_and_agents(x: chex.Array) -> chex.Array:
    """Concatenates the time and agent dimensions in the input tensor.

    Args:
    ----
        x: Input tensor of shape (Batch, Agents, Time, ...).

    Returns:
    -------
        chex.Array: Tensor of shape (Batch, Time x Agents, ...).
    """
    x = jnp.moveaxis(x, 0, 1)
    x = jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:]))
    return x


def get_init_hidden_state(actor_net_config: DictConfig, batch_size: int) -> HiddenStates:
    """Initializes the hidden states for the encoder and decoder."""
    # Compute the hidden state size based on embedding dimension and number of heads
    hidden_size = actor_net_config.embed_dim // actor_net_config.n_head

    # Define the shape of the hidden states
    hidden_state_shape = (
        batch_size,
        actor_net_config.n_head,
        actor_net_config.n_block,
        hidden_size,
        hidden_size,
    )

    # Initialize hidden states for encoder and decoder
    dec_hs_self_retm = jnp.zeros(hidden_state_shape)
    dec_hs_cross_retn = jnp.zeros(hidden_state_shape)
    enc_hs = jnp.zeros(hidden_state_shape)
    hidden_states = HiddenStates(
        encoder_hstate=enc_hs, decoder_hstate=(dec_hs_self_retm, dec_hs_cross_retn)
    )
    return hidden_states


class PositionalEncoding(nn.Module):
    """Positional Encoding for Sable. Encodes position information into sequences"""
    net_config: DictConfig
    d_model: int

    def setup(self) -> None:
        # Set maximum sequence length for positional encoding
        self.max_size = 10_000
        # Precompute the scaling factor for even indices (used in sine and cosine functions)
        self.div_term = jnp.exp(
            jnp.arange(0, self.d_model, 2) * (-jnp.log(10000.0) / self.d_model)
        )[jnp.newaxis]
        # Add a flag to enable positional encoding based on the network type
        if self.net_config.type == "rec_sable":
            self.do_pos_enc = self.net_config.timestep_positional_encoding
        else:
            self.do_pos_enc = self.agents_positional_encoding

    def __call__(self, key: chex.Array, query: chex.Array, value: chex.Array, position: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        """Computes positional encoding for a given sequence of positions."""
        # Check if positional encoding is enabled
        if self.do_pos_enc:
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
