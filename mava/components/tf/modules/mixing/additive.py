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

import tensorflow as tf
from tensorflow import Tensor

from mava.components.tf.architectures.base import BaseArchitecture
from mava.components.tf.modules.mixing.base import BaseMixingModule


class AdditiveMixing(BaseMixingModule):
    """Multi-agent monotonic mixing architecture."""

    def __init__(self, architecture: BaseArchitecture) -> None:
        """Initializes the mixer."""
        super(AdditiveMixing, self).__init__()

    def __call__(self, q_values: Tensor) -> Tensor:
        """Monotonic mixing logic."""
        # Not sure if this is the way to simply sum in tf.
        # I'm looking for an equivalent to th.sum(...) in PyTorch.
        return tf.math.accumulate_n(q_values)
        # or reduce_sum() along agent dim? Depends on q_values.shape
