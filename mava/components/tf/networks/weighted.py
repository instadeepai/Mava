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

# TODO (StJohn): complete class for weighted mixing
# https://github.com/oxwhirl/wqmix/tree/master/src/modules/mixers
"""Mixing for multi-agent RL systems"""

import sonnet as snt
import tensorflow as tf


class WeightedMixing(snt.Module):
    """Multi-agent mixing architecture."""

    def __init__(self) -> None:
        return

    def __call__(self) -> tf.Tensor:
        """Perform some mixing logic"""
        return tf.constant(1)
