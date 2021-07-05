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
from typing import Dict, Mapping, Sequence, Union

from acme import types

from mava import specs as mava_specs
from mava.systems.tf.madqn.networks import (
    make_default_networks as make_default_networks_madqn,
)
from mava.utils.enums import ArchitectureType, Network


# Default networks for qmix
def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (128, 128),
    shared_weights: bool = True,
    archecture_type: ArchitectureType = ArchitectureType.feedforward,
    network_type: Network = Network.mlp,
    fingerprints: bool = False,
) -> Mapping[str, types.TensorTransformation]:
    """Default networks for qmix.

    Args:
        environment_spec (mava_specs.MAEnvironmentSpec): description of the action and
            observation spaces etc. for each agent in the system.
        policy_networks_layer_sizes (Union[Dict[str, Sequence], Sequence], optional):
            size of policy networks. Defaults to (128,128).
        shared_weights (bool, optional): whether agents should share weights or not.
            Defaults to True.
        archecture_type (ArchitectureType, optional): archecture used for
            agent networks. Can be feedforward or recurrent.
            Defaults to ArchitectureType.recurrent.
        network_type (Network, optional): Agent network type. Can be mlp,
            atari_dqn_network or coms_network. Defaults to Network.coms_network.
        fingerprints (bool, optional): whether to apply replay stabilisation using
            policy fingerprints. Defaults to False.

    Returns:
        Mapping[str, types.TensorTransformation]: returned agent networks.
    """

    return make_default_networks_madqn(
        environment_spec=environment_spec,
        policy_networks_layer_sizes=policy_networks_layer_sizes,
        shared_weights=shared_weights,
        archecture_type=archecture_type,
        network_type=network_type,
        fingerprints=fingerprints,
    )
