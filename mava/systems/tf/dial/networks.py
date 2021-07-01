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
from typing import Mapping

from acme import types

from mava import specs as mava_specs
from mava.systems.tf.madqn.networks import (
    make_default_networks as make_default_networks_madqn,
)
from mava.utils.enums import ArchitectureType, Network


# Default networks for Dial
def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    message_size: int = 1,
    shared_weights: bool = True,
    archecture_type: ArchitectureType = ArchitectureType.recurrent,
    network_type: Network = Network.coms_network,
    fingerprints: bool = False,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""

    assert (
        archecture_type == ArchitectureType.recurrent
    ), "Dial currently only supports recurrent architectures."

    return make_default_networks_madqn(
        environment_spec=environment_spec,
        shared_weights=shared_weights,
        archecture_type=archecture_type,
        network_type=network_type,
        fingerprints=fingerprints,
        message_size=message_size,
    )
