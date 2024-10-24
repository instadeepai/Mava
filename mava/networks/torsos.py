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

    def setup(self) -> None:
        self.activation_fn = _parse_activation_fn(self.activation)

    @nn.compact
    def __call__(self, observation: chex.Array) -> chex.Array:
        """Forward pass."""
        x = observation
        for layer_size in self.layer_sizes:
            x = nn.Dense(layer_size, kernel_init=orthogonal(np.sqrt(2)))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            x = self.activation_fn(x)
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
        for channel, kernel, stride in zip(self.channel_sizes, self.kernel_sizes, self.strides):
            x = nn.Conv(channel, (kernel, kernel), (stride, stride))(x)
            if self.use_layer_norm:
                x = nn.LayerNorm(use_scale=False)(x)
            x = self.activation_fn(x)

        # Collapse (merge) the last three dimensions (width, height, channels)
        # Leave the batch, agent and time (if recurrent) dims unchanged.
        return jax.lax.collapse(x, -3)


class SwiGLU(nn.Module):
    ffn_dim: int
    embed_dim: int

    def setup(self) -> None:
        self.W_1 = self.param("W_1", nn.initializers.zeros, (self.embed_dim, self.ffn_dim))
        self.W_G = self.param("W_G", nn.initializers.zeros, (self.embed_dim, self.ffn_dim))
        self.W_2 = self.param("W_2", nn.initializers.zeros, (self.ffn_dim, self.embed_dim))

    def __call__(self, x: chex.Array) -> chex.Array:
        return (jax.nn.swish(x @ self.W_G) * (x @ self.W_1)) @ self.W_2


def _parse_activation_fn(activation_fn_name: str) -> Callable[[chex.Array], chex.Array]:
    """Get the activation function."""
    activation_fns: Dict[str, Callable[[chex.Array], chex.Array]] = {
        "relu": nn.relu,
        "tanh": nn.tanh,
    }
    return activation_fns[activation_fn_name]
