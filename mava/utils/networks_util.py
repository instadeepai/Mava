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
