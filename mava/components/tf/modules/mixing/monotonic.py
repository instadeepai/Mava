# python3
# Copyright 2021 InstaDeep Ltd. All rights reserved.
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

# TODO (StJohn): complete class for monotonic mixing

"""Mixing for multi-agent RL systems"""

from typing import Dict

import tensorflow as tf
from tensorflow import Tensor

from mava.components.tf.architectures import BaseArchitecture
from mava.components.tf.modules.mixing import BaseMixingModule


class MonotonicMixing(BaseMixingModule):
    """Multi-agent mixing architecture."""

    def __init__(
        self,
        architecture: BaseArchitecture,
        mixer: str = "vdn",
    ) -> None:
        """Initializes the mixer.
        Args:
            architecture: the BaseArchitecture used.
            mixer: the type of monotonic mixing.
        """
        self._architecture = architecture
        self._mixer = mixer.lower()

    def forward(
        self,
        agent_qs: Dict[str, float],  # Check type
        states: Dict[str, float],  # Check type
        num_hypernet_layers: int = 1,
    ) -> Tensor:

        """Monotonic mixing logic."""
        if self._mixer == "vdn":
            # Not sure if this is the way to simply sum in tf.
            # I'm looking for an equivalent to th.sum(...) in PyTorch.
            return tf.math.accumulate_n(agent_qs)

        elif self._mixer == "qmix":
            # Set up hypernetwork configuration
            if num_hypernet_layers == 1:
                # Create 1-layer NN
                pass
            elif num_hypernet_layers == 2:
                # Create 2-layer NN
                pass
            elif num_hypernet_layers > 2:
                raise Exception("Sorry >2 hypernet layers is not implemented!")
            else:
                raise Exception("Error setting number of hypernet layers.")

            # State dependent bias for hidden layer

            # # V(s) instead of a bias for the last layers

            # Forward pass
