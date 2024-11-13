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

from typing import Callable, Dict, Sequence

import chex
import jax
import numpy as np
from flax import linen as nn
from flax.linen.initializers import orthogonal


class MLPTorso(nn.Module):
    """MLP torso."""

    layer_sizes: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False
    activate_final: bool = True

    def setup(self) -> None:
        self.activation_fn = _parse_activation_fn(self.activation)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for i, layer_size in enumerate(self.layer_sizes):
            x = nn.Dense(layer_size, kernel_init=orthogonal(np.sqrt(2)))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)

            should_activate = (i < len(self.layer_sizes) - 1) or self.activate_final
            x = self.activation_fn(x) if should_activate else x

        return x


class CNNTorso(nn.Module):
    """CNN torso."""

    channel_sizes: Sequence[int]
    kernel_sizes: Sequence[int]
    strides: Sequence[int]
    activation: str = "relu"
    use_layer_norm: bool = False

    def setup(self) -> None:
        self.activation_fn = _parse_activation_fn(self.activation)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for channel, kernel, stride in zip(
            self.channel_sizes, self.kernel_sizes, self.strides, strict=True
        ):
            x = nn.Conv(channel, (kernel, kernel), (stride, stride))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            x = self.activation_fn(x)

        # Collapse (merge) the last three dimensions (width, height, channels)
        # Leave the batch, agent and time (if recurrent) dims unchanged.
        return jax.lax.collapse(x, -3)


class SwiGLU(nn.Module):
    """SwiGLU module.
    A gated variation of a standard feedforward layer using a Swish activation function.
    For more details see: https://arxiv.org/abs/2002.05202
    """

    hidden_dim: int
    embed_dim: int

    def setup(self) -> None:
        self.W_linear = self.param(
            "W_linear", nn.initializers.zeros, (self.embed_dim, self.hidden_dim)
        )
        self.W_gate = self.param("W_gate", nn.initializers.zeros, (self.embed_dim, self.hidden_dim))
        self.W_output = self.param(
            "W_output", nn.initializers.zeros, (self.hidden_dim, self.embed_dim)
        )

    def __call__(self, x: chex.Array) -> chex.Array:
        gated_output = jax.nn.swish(x @ self.W_gate) * (x @ self.W_linear)
        return gated_output @ self.W_output


def _parse_activation_fn(activation_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    """Get the activation function."""
    activation_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "relu": nn.relu,
        "tanh": nn.tanh,
    }
    return activation_fns[activation_fn_name]
