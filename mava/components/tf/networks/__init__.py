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


from acme.tf.networks.atari import DQNAtariNetwork
from acme.tf.networks.distributional import MultivariateNormalDiagHead
from acme.tf.networks.multiplexers import CriticMultiplexer
from acme.tf.networks.noise import ClippedGaussian
from acme.tf.networks.rescaling import ClipToSpec, RescaleToSpec, TanhToSpec

from mava.components.tf.networks.communication import CommunicationNetwork
from mava.components.tf.networks.continuous import (
    LayerNormAndResidualMLP,
    LayerNormMLP,
    NearZeroInitializedLinear,
)
from mava.components.tf.networks.convolution import Conv1DNetwork
from mava.components.tf.networks.distributional import CategoricalHead
from mava.components.tf.networks.mad4pg import (
    DiscreteValuedDistribution,
    DiscreteValuedHead,
)
