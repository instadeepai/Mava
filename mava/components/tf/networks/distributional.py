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


from typing import Any, List, Optional, Union

import numpy as np
import sonnet as snt
import tensorflow as tf
from tensorflow_probability import distributions as tfd


# Based on
# https://github.com/deepmind/acme/blob/6bf350df1d9dd16cd85217908ec9f47553278976/acme/jax/networks/distributional.py#L33 # noqa: E501
class CategoricalHead(snt.Module):
    """Module that produces a categorical distribution for a given number of values."""

    def __init__(
        self,
        num_values: Union[int, List[int]],
        dtype: Optional[Any] = np.int32,
        w_init: Optional[snt.initializers.Initializer] = None,
        b_init: Optional[snt.initializers.Initializer] = None,
        name: Optional[str] = None,
    ):
        """Init.

        Args:
            num_values : number of values for categorical distribution.
            dtype : type for returned dist.
            w_init: Initialization for linear layer weights.
            b_init: Initialization for linear layer biases.
            name : name for module.
        """
        super().__init__(name=name)
        self._dtype = dtype
        self._logit_shape = num_values
        self._linear = snt.Linear(np.prod(num_values), w_init=w_init, b_init=b_init)

    def __call__(self, inputs: tf.Tensor) -> tfd.Distribution:
        """Pass inputs through networks.

        Args:
            inputs : intermediate inputs of network.

        Returns:
            categorical distribution.
        """
        logits = self._linear(inputs)
        if not isinstance(self._logit_shape, int):
            logits = tf.reshape(logits, self._logit_shape)
        return tfd.Categorical(logits=logits, dtype=self._dtype)
