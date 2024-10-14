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
from flax import linen as nn

# TODO: update this
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

from functools import partial

import chex
import jax
import jax.numpy as jnp


def concat_time_and_agents(x: chex.Array):
    """Concatenate the time and agent dimensions."""
    x = jnp.moveaxis(x, 0, 1)
    x = jnp.reshape(x, (x.shape[0], x.shape[1] * x.shape[2], *x.shape[3:]))
    return x


def revert_concatenation(x: chex.Array, num_agents: int):
    """Revert the concatenation of the time and agent dimensions."""
    x = jnp.reshape(x, (x.shape[0], x.shape[1] // num_agents, num_agents, *x.shape[2:]))
    x = jnp.moveaxis(x, 1, 0)
    return x


@partial(jax.jit, static_argnames=("n_agents",))
def duplicate_rows_columns_jax(matrix: chex.Array, n_agents: int) -> chex.Array:
    """Duplicate rows and columns of a matrix based on the number of agents."""
    # Duplicate columns
    cols = matrix.shape[1]
    col_indices = jnp.repeat(jnp.arange(cols), n_agents)
    matrix_col_duplicated = matrix[:, col_indices]

    # Duplicate rows
    rows = matrix_col_duplicated.shape[0]
    row_indices = jnp.repeat(jnp.arange(rows), n_agents)
    matrix_both_duplicated = matrix_col_duplicated[row_indices, :]

    return matrix_both_duplicated


@partial(jax.jit, static_argnames=("n_agents",))
def duplicate_rows_jax(matrix: chex.Array, n_agents: int) -> chex.Array:
    """Duplicate rows of a matrix based on the number of agents."""
    # Duplicate rows
    rows = matrix.shape[0]
    row_indices = jnp.repeat(jnp.arange(rows), n_agents)
    matrix_row_duplicated = matrix[row_indices, :]

    return matrix_row_duplicated


@partial(jax.jit, static_argnames=("steps", "decay_kappa"))
def create_decay_tril_matrix_with_restarts(
    steps: int, dones: chex.Array, decay_kappa: int
) -> chex.Array:
    """Create a lower triangular matrix with decay and restarts where the episode is ended."""

    # Cumulative sum to get section identifiers
    restarts = jnp.cumsum(dones)

    # Create meshgrids for row and column restart identifiers and for decay exponent calculation
    row_restarts, col_restarts = jnp.meshgrid(restarts, restarts, indexing="ij")
    row_indices, col_indices = jnp.meshgrid(
        jnp.arange(steps), jnp.arange(steps), indexing="ij"
    )

    # Calculate decay exponents within each section
    exponents = row_indices - col_indices
    # Generate mask for valid lower triangular elements that belong to the same section
    same_section_mask = row_restarts == col_restarts
    lower_triangular_mask = row_indices >= col_indices

    # Apply decay where valid, within the same section and lower triangular
    valid_mask = same_section_mask & lower_triangular_mask
    decay_matrix = jnp.zeros((steps, steps))
    decay_matrix = jnp.where(valid_mask, decay_kappa**exponents, 0)

    return decay_matrix


@partial(jax.jit, static_argnames=("steps", "decay_kappa", "n_agents"))
def get_decay_matrices(
    steps: int, dones: chex.Array, decay_kappa: float, n_agents: int
) -> chex.Array:
    """Get the encoder and decoder decay matrices."""
    encoder_decay_matrix_one_agent = create_decay_tril_matrix_with_restarts(
        steps, dones, decay_kappa
    )
    encoder_decay_matrix = duplicate_rows_columns_jax(
        encoder_decay_matrix_one_agent, n_agents
    )

    mask_agents = jnp.tril(jnp.ones((steps * n_agents, steps * n_agents))).reshape(
        steps * n_agents, steps * n_agents
    )
    decoder_decay_matrix = mask_agents * encoder_decay_matrix
    return encoder_decay_matrix, decoder_decay_matrix, encoder_decay_matrix_one_agent


@partial(jax.jit, static_argnames=("decay_kappa", "n_agents"))
def get_inner_decay_matrices(
    decay_matrix_chunk: chex.Array,
    decay_matrix_one_agent: chex.Array,
    decay_kappa: float,
    n_agents: int,
) -> chex.Array:
    """Get the value and query inner decay matrices for the chunkwise representation."""
    # Value inner decay: (bacth, n_heads, chunk_size, 1)
    value_inner_decay = jnp.expand_dims(decay_matrix_chunk[-1], -1)

    # Query inner decay: (batch, n_heads, chunk_size, 1)
    last_row = decay_matrix_one_agent[-1]
    non_zero_mask = last_row > 0
    query_inner_decay = jnp.cumsum(non_zero_mask)
    query_inner_decay = decay_kappa ** jnp.where(
        non_zero_mask, query_inner_decay, jnp.inf
    )
    query_inner_decay = jnp.expand_dims(query_inner_decay, -1)
    query_inner_decay = duplicate_rows_jax(query_inner_decay, n_agents)
    return value_inner_decay, query_inner_decay


def shift_actions(actions, shift_params):
    """Shift the actions to create a new array with an additional item."""
    batch_size, action_dim, n_agents = shift_params
    one_hot_action = jax.nn.one_hot(actions, action_dim)  # (batch, n_agents, action_dim)
    shifted_actions = jnp.zeros((batch_size, n_agents, action_dim + 1))
    shifted_actions = shifted_actions.at[:, 0, 0].set(1)
    # This should look like this for all batches:
    # [[1, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0],
    #  [0, 0, 0, 0, 0, 0]]
    shifted_actions = shifted_actions.at[:, 1:, 1:].set(one_hot_action[:, :-1, :])
    # If the actions are: [2, 1, 0]
    # The one hot action is:
    # [[0, 0, 1, 0, 0],
    #  [0, 1, 0, 0, 0],
    #  [1, 0, 0, 0, 0]]

    # The shifted action will be:
    # [[1, 0, 0, 0, 0, 0],   Agent0 has no previous agent
    #  [0, 0, 0, 1, 0, 0],
    #  [0, 0, 1, 0, 0, 0]]
    return shifted_actions


def get_init_hidden_state(config, batch_size):
    hidden_size = (
        config.network.actor_network.embed_dim // config.network.actor_network.n_head
    )
    hidden_state_shape = (
        batch_size,
        config.network.actor_network.n_head,
        config.network.actor_network.n_block,
        hidden_size,
        hidden_size,
    )
    dec_hs_self_attn = jnp.zeros(hidden_state_shape)
    dec_hs_cross_attn = jnp.zeros(hidden_state_shape)
    enc_hs = jnp.zeros(hidden_state_shape)

    return enc_hs, (dec_hs_self_attn, dec_hs_cross_attn)


if __name__ == "__main__":
    steps = 5
    n_agents = 2
    decay_kappa = 0.5
    dones = jnp.array([False, False, True, False, False])
    encoder_decay_matrix, decoder_decay_matrix, encoder_decay_matrix_one_agent = (
        get_decay_matrices(steps, dones, decay_kappa, n_agents)
    )
    print("encoder_decay_matrix: \n", encoder_decay_matrix)
    print("decoder_decay_matrix: \n", decoder_decay_matrix)
