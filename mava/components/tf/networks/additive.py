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

import sonnet as snt
import tensorflow as tf


class AdditiveMixingNetwork(snt.Module):
    """Multi-agent monotonic mixing architecture."""

    def __init__(self, name: str = "mixing") -> None:
        """Initializes the mixer."""
        super(AdditiveMixingNetwork, self).__init__(name=name)

    def __call__(self, q_values: tf.Tensor) -> tf.Tensor:
        """Monotonic mixing logic."""
        # return tf.math.reduce_sum(q_values, axis=1)
        return tf.math.reduce_sum(q_values, axis=1)
