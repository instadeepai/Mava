# Copyright 2022 InstaDeep Ltd. All rights areserverd.
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

from typing import Sequence

import chex
import jax.numpy as jnp
from flax import linen as nn

from mava.networks.encoders.base import Encoder


class RwareEncoder(Encoder):
    """Encoder for the Rware environment."""

    def __call__(self, observation: chex.ArrayTree) -> chex.Array:
        """Forward pass of the encoder."""
        return observation.agents_view.astype(jnp.float32)
