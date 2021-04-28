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
from acme.tf.networks import DQNAtariNetwork

from mava.types import OLT


class MADQNAtariNetwork(DQNAtariNetwork):
    """A feed-forward network for use with Ape-X DQN.
    See https://arxiv.org/pdf/1803.00933.pdf for more information.
    """

    def __init__(self, num_actions: int):
        super().__init__(num_actions)

    def __call__(self, inputs: OLT) -> tf.Tensor:
        return self._network(inputs.observation)
