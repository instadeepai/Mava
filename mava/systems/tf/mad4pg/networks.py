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
from typing import Dict, Mapping, Optional, Sequence, Union

from acme import types
from dm_env import specs

from mava import specs as mava_specs
from mava.systems.tf.maddpg.networks import (
    make_default_networks as make_default_networks_maddpg,
)
from mava.utils.enums import ArchitectureType

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    vmin: float = None,
    vmax: float = None,
    net_spec_keys: Dict[str, str] = {},
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = None,
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    sigma: float = 0.3,
    archecture_type: ArchitectureType = ArchitectureType.feedforward,
    num_atoms: int = 51,
    seed: Optional[int] = None,
) -> Mapping[str, types.TensorTransformation]:
    """Default networks for mad4pg.

    Args:
        environment_spec: description of the action and
            observation spaces etc. for each agent in the system.
        agent_net_keys: specifies what network each agent uses.
        vmin: hyperparameters for the distributional critic in mad4pg.
        vmax: hyperparameters for the distributional critic in mad4pg.
        net_spec_keys: specifies the specs of each network.
        policy_networks_layer_sizes: size of policy networks.
        critic_networks_layer_sizes: size of critic networks.
        sigma: hyperparameters used to add Gaussian noise
            for simple exploration. Defaults to 0.3.
        archecture_type: archecture used
            for agent networks. Can be feedforward or recurrent.
            Defaults to ArchitectureType.feedforward.

        num_atoms:  hyperparameters for the distributional critic in
            mad4pg.
        seed: random seed for network initialization.

    Returns:
        returned agent networks.
    """
    if not vmin or not vmax:
        raise ValueError(
            "vmin and vmax cannot be None. They should be set to the"
            "minimum and maximum cumulative reward in the environment."
        )
    return make_default_networks_maddpg(
        environment_spec=environment_spec,
        agent_net_keys=agent_net_keys,
        net_spec_keys=net_spec_keys,
        policy_networks_layer_sizes=policy_networks_layer_sizes,
        critic_networks_layer_sizes=critic_networks_layer_sizes,
        sigma=sigma,
        archecture_type=archecture_type,
        vmin=vmin,
        vmax=vmax,
        num_atoms=num_atoms,
        seed=seed,
    )
