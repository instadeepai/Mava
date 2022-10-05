# python3
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

from typing import Any, Optional, Tuple

import haiku as hk
import jax.numpy as jnp
import tensorflow_probability

tfp = tensorflow_probability.substrates.jax
tfd = tfp.distributions


class CategoricalValueHead(hk.Module):
    """Network head that produces a categorical distribution and value.

    Similar to https://github.com/deepmind/acme/blob/master/acme/jax/networks/distributional.py#L289, # noqa: E501
    except you can set the dtype of the distribution.
    """

    def __init__(
        self,
        num_values: int,
        name: Optional[str] = None,
        dtype: Optional[Any] = jnp.int32,
    ):
        """Init Module.

        Args:
            num_values : num values for dist to output.
            name : name for module.
            dtype : dtype for module.
        """
        super().__init__(name=name)
        self._dtype = dtype
        self._logit_layer = hk.Linear(num_values)
        self._value_layer = hk.Linear(1)

    def __call__(self, inputs: jnp.ndarray) -> Tuple[tfd.Distribution, jnp.float32]:  # type: ignore # noqa: E501
        """Forward pass.

        Args:
            inputs : input data.

        Returns:
            distribution and value.
        """
        logits = self._logit_layer(inputs)
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return (tfd.Categorical(logits=logits, dtype=self._dtype), value)
