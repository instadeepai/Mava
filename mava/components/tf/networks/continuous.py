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
"""Networks used in continous settings.

Some networks adapted from
https://github.com/deepmind/acme/blob/master/acme/tf/networks/continuous.py.
"""
from typing import Any, Optional, Sequence

import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf.networks.continuous import ResidualLayernormWrapper


def get_initialization(seed: Optional[int] = None) -> Any:
    """Funtion that returns seeded initialization.

    Args:
        seed (Optional[int], optional): seed used for initialization.
            Defaults to None.

    Returns:
        Any: returned initializers.
    """
    return tf.initializers.VarianceScaling(
        distribution="uniform", mode="fan_out", scale=0.333, seed=seed
    )


class NearZeroInitializedLinear(snt.Linear):
    """Simple linear layer, initialized at near zero weights and zero biases."""

    def __init__(
        self, output_size: int, scale: float = 1e-4, seed: Optional[int] = None
    ):
        """Constructor for seeded initializer.

        Args:
            output_size (int): output size of layer.
            scale (float, optional): scale for variance scaling. Defaults to 1e-4.
            seed (Optional[int], optional): seed for initialization. Defaults to None.
        """
        super().__init__(
            output_size, w_init=tf.initializers.VarianceScaling(scale, seed=seed)
        )


class LayerNormMLP(snt.Module):
    """Simple feedforward MLP torso with optional initial layer-norm."""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        activate_final: bool = False,
        intermediate_act: tf.function = tf.nn.tanh,
        final_act: tf.function = tf.nn.elu,
        layernorm: bool = True,
        seed: int = None,
    ):
        """Constructs MLP.

        Args:
            layer_sizes (Sequence[int]): a sequence of ints specifying the size of each
                layer.
            activate_final (bool, optional): whether or not to use an activation
                function on the final layer of the neural network.
            intermediate_act (tf.function, optional): activation function for
                intermediate layers.Defaults to tf.nn.tanh.
            final_act (tf.function, optional): activation function for final layer.
                Defaults to tf.nn.elu.
            layernorm (bool, optional): whether to apply layernorm. Defaults to True.
            seed (int, optional): random seed used to initialize networks.
                Defaults to None.
        """
        super().__init__(name="feedforward_mlp_torso")

        self._seed = seed
        # First Layer
        network = [snt.Linear(layer_sizes[0], w_init=get_initialization(self._seed))]

        if layernorm:
            network += [
                snt.LayerNorm(
                    axis=slice(1, None), create_scale=True, create_offset=True
                )
            ]

        # Intermediate to final layer
        network += [
            intermediate_act,
            snt.nets.MLP(
                layer_sizes[1:],
                w_init=get_initialization(self._seed),
                activation=final_act,
                activate_final=activate_final,
            ),
        ]

        self._network = snt.Sequential(network)

    def __call__(self, observations: types.Nest) -> tf.Tensor:
        """Forward for the policy network."""
        return self._network(tf2_utils.batch_concat(observations))


class LayerNormAndResidualMLP(snt.Module):
    """MLP with residual connections and layer norm.

    An MLP which applies residual connection and layer normalisation every two
    linear layers. Similar to Resnet, but with FC layers instead of convolutions.
    """

    def __init__(self, hidden_size: int, num_blocks: int, seed: int = None):
        """Create the model.

        Args:
            hidden_size (int): width of each hidden layer.
            num_blocks (int): number of blocks, each block being MLP([hidden_size,
                hidden_size]) + layer norm + residual connection.
            seed (int, optional): random seed used to initialize
                networks. Defaults to None.
        """
        super().__init__(name="LayerNormAndResidualMLP")

        self._seed = seed

        # Create initial MLP layer.
        layers = [snt.nets.MLP([hidden_size], w_init=get_initialization(self._seed))]

        # Follow it up with num_blocks MLPs with layernorm and residual connections.
        for _ in range(num_blocks):
            mlp = snt.nets.MLP(
                [hidden_size, hidden_size], w_init=get_initialization(self._seed)
            )
            layers.append(ResidualLayernormWrapper(mlp))

        self._network = snt.Sequential(layers)

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        """Forward for the policy network."""
        return self._network(inputs)
