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

# based on the following:
# https://www.tensorflow.org/addons/api_docs/python/tfa/layers/NoisyDense

"""Adaptation of Keras Noisy Dense Layer using Sonnet Base Module"""

from typing import Optional, Sequence, Any

import sonnet as snt

import math
import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import utils as tf2_utils
from sonnet.src import base
from sonnet.src import initializers
from sonnet.src import once
from sonnet.src import utils


def _scaled_noise(size, dtype):
    x = tf.random.normal(shape=size, dtype=dtype)
    return tf.sign(x) * tf.sqrt(tf.abs(x))


class NoisyLinear(base.Module):
    """Noisy Linear module, optionally including bias."""

    def __init__(self,
                 output_size: int,
                 sigma: float = 0.5,
                 use_factorised: bool = True,
                 with_bias: bool = True,
                 w_init: Optional[initializers.Initializer] = None,
                 b_init: Optional[initializers.Initializer] = None,
                 name: Optional[str] = None):
        """Constructs a `Noisy Linear` module.
    Args:
        output_size: Output dimensionality.
        sigma: A float between 0-1 used as a standard deviation figure and is
            applied to the gaussian noise layer (`sigma_kernel` and `sigma_bias`). 
            (uses only if use_factorised=True)
        use_factorised: Boolean, whether the layer uses independent or 
            factorised Gaussian noise
        with_bias: Whether to include bias parameters. Default `True`.
        w_init: Optional initializer for the weights. By default the weights are
        initialized truncated random normal values with a standard deviation of
        `1 / sqrt(input_feature_size)`, which is commonly used when the inputs
        are zero centered (see https://arxiv.org/abs/1502.03167v3).
        b_init: Optional initializer for the bias. By default the bias is
        initialized to zero.
        name: Name of the module.
    """
        super().__init__(name=name)
        self.output_size = output_size
        self.sigma = sigma
        self.use_factorised = use_factorised
        self.with_bias = with_bias
        self.w_init = w_init
        if with_bias:
            self.b_init = b_init if b_init is not None else initializers.Zeros()
        elif b_init is not None:
            raise ValueError("When not using a bias the b_init must be None.")

    @once.once
    def _initialize(self, inputs: tf.Tensor):
        """Constructs parameters used by this module."""
        utils.assert_minimum_rank(inputs, 2)

        self.dtype = inputs.dtype

        input_size = inputs.shape[-1]
        if input_size is None:  # Can happen inside an @tf.function.
            raise ValueError("Input size must be specified at module build time.")

        self.input_size = input_size

        if self.w_init is None:
            # See https://arxiv.org/abs/1502.03167v3.
            stddev = 1 / math.sqrt(self.input_size)
            self.w_init = initializers.TruncatedNormal(stddev=stddev)

        self.w = tf.Variable(
            self.w_init([self.input_size, self.output_size], inputs.dtype),
            name="w", trainable=True)

        sqrt_dim = self.input_size ** (1 / 2)
        if self.input_size is None:
            raise ValueError(
                "The last dimension of the inputs to `Dense` "
                "should be defined. Found `None`."
            )

        # use factorising Gaussian variables
        if self.use_factorised:
            sigma_init = self.sigma / sqrt_dim
        # use independent Gaussian variables
        else:
            sigma_init = 0.017

        sigma_init = initializers.Constant(value=sigma_init)
        eps_init = initializers.Zeros()

        self.sigma_kernel = tf.Variable(sigma_init(
            [self.input_size, self.output_size], inputs.dtype), name="sigma_kernel", trainable=True)
        self.eps_kernel = tf.Variable(eps_init(
            [self.input_size, self.output_size], inputs.dtype), name="eps_kernel", trainable=False)

        if self.with_bias:
            self.b = tf.Variable(
                self.b_init([self.output_size], inputs.dtype), name="b", trainable=True)

            self.sigma_bias = tf.Variable(sigma_init(
                [self.output_size, ], inputs.dtype), name="sigma_bias", trainable=True)

            self.eps_bias = tf.Variable(
                eps_init([self.output_size, ], inputs.dtype), name="eps_bias", trainable=False)
        else:
            self.sigma_bias = None
            self.eps_bias = None

        self.reset_noise()

    def __call__(self, inputs: tf.Tensor) -> tf.Tensor:
        self._initialize(inputs)

        outputs = tf.matmul(inputs, tf.add(
            self.w, tf.multiply(self.sigma_kernel, self.eps_kernel)))
        if self.with_bias:
            outputs = tf.add(outputs, tf.add(
                self.b, tf.multiply(self.sigma_bias, self.eps_bias)))
        return outputs

    def reset_noise(self):
        """Create the factorised Gaussian noise."""

        if self.use_factorised:
            # Generate random noise
            in_eps = _scaled_noise([self.input_size, 1], dtype=self.dtype)
            out_eps = _scaled_noise([1, self.output_size], dtype=self.dtype)

            # Scale the random noise
            self.eps_kernel.assign(tf.matmul(in_eps, out_eps))
            self.eps_bias.assign(out_eps[0])
        else:
            # generate independent variables
            self.eps_kernel.assign(
                tf.random.normal(
                    shape=[self.input_size, self.output_size], dtype=self.dtype)
            )
            self.eps_bias.assign(
                tf.random.normal(
                    shape=[
                        self.output_size,
                    ],
                    dtype=self.dtype,
                )
            )

    def remove_noise(self):
        """Remove the factorised Gaussian noise."""

        self.eps_kernel.assign(
            tf.zeros([self.input_size, self.output_size], dtype=self.dtype))
        self.eps_bias.assign(tf.zeros([self.output_size], dtype=self.dtype))


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


class NoisyMLP(snt.Module):
    """Simple feedforward Noisy MLP torso - the first layer is not noisy and each layer after that is a noisy linear layer"""

    def __init__(
        self,
        layer_sizes: Sequence[int],
        sigma: float = 0.5,
        use_factorised: bool = True,
        activate_final: bool = False,
        intermediate_act: tf.function = tf.nn.tanh,
        final_act: tf.function = tf.nn.elu,
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
            seed (int, optional): random seed used to initialize networks.
                Defaults to None.
        """
        super().__init__(name="feedforward_noisy_mlp_torso")

        self._seed = seed

        # First Layer
        network = [snt.Linear(layer_sizes[0], w_init=get_initialization(self._seed))]

        for index, output_size in enumerate(layer_sizes[1:]):
            network.append(intermediate_act)
            network.append(
                NoisyLinear(
                    output_size=output_size,
                    sigma=sigma,
                    use_factorised=use_factorised,
                    w_init=get_initialization(self._seed),
                    name="noisy_linear_%d" % index))

        if activate_final:
            network.append(final_act)

        self._network = snt.Sequential(network)

    def __call__(self, observations: types.Nest) -> tf.Tensor:
        """Forward for the policy network."""
        return self._network(tf2_utils.batch_concat(observations))

    def reset_noise(self):
        for layer in self._network._layers[1:]:
            if isinstance(layer, NoisyLinear):
                layer.reset_noise()

    def remove_noise(self):
        for layer in self._network._layers[1:]:
            if isinstance(layer, NoisyLinear):
                layer.remove_noise()
