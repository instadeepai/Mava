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

"""Implementation of an mlp module with layer normalisation."""
# Adapted from https://github.com/deepmind/dm-haiku/blob/main/haiku/_src/nets/mlp.py

from typing import Any, Callable, Iterable, Optional

import haiku as hk
import jax
import jax.numpy as jnp


class MLP_NORM(hk.Module):
    """A multi-layer perceptron module with an option for layer normalisation."""

    def __init__(
        self,
        output_sizes: Iterable[int],
        w_init: Optional[hk.initializers.Initializer] = None,
        b_init: Optional[hk.initializers.Initializer] = None,
        with_bias: bool = True,
        activation: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
        activate_final: bool = False,
        layer_norm: bool = False,
        name: Optional[str] = None,
    ):
        """Constructs an MLP with layer normalisation (MLP_NORM).
        Args:
            output_sizes: Sequence of layer sizes.
            w_init: Initializer for :class:`~haiku.Linear` weights.
            b_init: Initializer for :class:`~haiku.Linear` bias. Must be ``None`` if
            ``with_bias=False``.
            with_bias: Whether or not to apply a bias in each layer.
            activation: Activation function to apply between :class:`~haiku.Linear`
            layers. Defaults to ReLU.
            activate_final: Whether or not to activate the final layer of the MLP.
            layer_norm: apply layer normalisation to the hidden MLP layers.
            name: Optional name for this module.
        Raises:
            ValueError: If ``with_bias`` is ``False`` and ``b_init`` is not ``None``.
        """
        if not with_bias and b_init is not None:
            raise ValueError("When with_bias=False b_init must not be set.")

        super().__init__(name=name)
        self.with_bias = with_bias
        self.w_init = w_init
        self.b_init = b_init
        self.activation = activation
        self.activate_final = activate_final
        self.layer_norm = layer_norm
        layers = []
        norms = []
        output_sizes = tuple(output_sizes)
        if self.layer_norm:
            for index, output_size in enumerate(output_sizes):
                layers.append(
                    hk.Linear(
                        output_size=output_size,
                        w_init=w_init,
                        b_init=b_init,
                        with_bias=with_bias,
                        name="linear_%d" % index,
                    )
                )
                norms.append(
                    hk.LayerNorm(
                        axis=-1,
                        create_scale=True,
                        create_offset=True,
                        param_axis=-1,
                        use_fast_variance=False,
                        name="norm_%d" % index,
                    )
                )
            self.norms = tuple(norms)
        else:
            for index, output_size in enumerate(output_sizes):
                layers.append(
                    hk.Linear(
                        output_size=output_size,
                        w_init=w_init,
                        b_init=b_init,
                        with_bias=with_bias,
                        name="linear_%d" % index,
                    )
                )

        self.layers = tuple(layers)
        self.output_size = output_sizes[-1] if output_sizes else None

    def __call__(
        self,
        inputs: jnp.ndarray,
        dropout_rate: Optional[float] = None,
        rng: Any = None,
    ) -> jnp.ndarray:
        """Connects the module to some inputs.
        Args:
            inputs: A Tensor of shape ``[batch_size, input_size]``.
            dropout_rate: Optional dropout rate.
            rng: Optional RNG key. Require when using dropout.
        Returns:
            The output of the model of size ``[batch_size, output_size]``.
        """
        if dropout_rate is not None and rng is None:
            raise ValueError("When using dropout an rng key must be passed.")
        elif dropout_rate is None and rng is not None:
            raise ValueError("RNG should only be passed when using dropout.")

        rng = hk.PRNGSequence(rng) if rng is not None else None
        num_layers = len(self.layers)

        out = inputs
        for i, layer in enumerate(self.layers):
            out = layer(out)
            if i < (num_layers - 1) or self.activate_final:
                # Only perform dropout if we are activating the output.
                if dropout_rate is not None:
                    out = hk.dropout(next(rng), dropout_rate, out)  # type: ignore
                out = self.activation(out)
            if self.layer_norm:
                out = self.norms[i](out)

        return out
