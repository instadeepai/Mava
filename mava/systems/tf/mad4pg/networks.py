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
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = None,
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    sigma: float = 0.3,
    archecture_type: ArchitectureType = ArchitectureType.feedforward,
    vmin: float = -150.0,
    vmax: float = 150.0,
    num_atoms: int = 51,
    seed: Optional[int] = None,
) -> Mapping[str, types.TensorTransformation]:
    """Default networks for mad4pg.

    Args:
        environment_spec (mava_specs.MAEnvironmentSpec): description of the action and
            observation spaces etc. for each agent in the system.
        policy_networks_layer_sizes (Union[Dict[str, Sequence], Sequence], optional):
            size of policy networks.
        critic_networks_layer_sizes (Union[Dict[str, Sequence], Sequence], optional):
            size of critic networks. Defaults to (512, 512, 256).
        shared_weights (bool, optional): whether agents should share weights or not.
            Defaults to True.
        sigma (float, optional): hyperparameters used to add Gaussian noise
            for simple exploration. Defaults to 0.3.
        archecture_type (ArchitectureType, optional): archecture used
            for agent networks. Can be feedforward or recurrent.
            Defaults to ArchitectureType.feedforward.
        vmin (float, optional): hyperparameters for the distributional critic in mad4pg.
            Defaults to -150.0.
        vmax (float, optional): hyperparameters for the distributional critic in mad4pg.
            Defaults to 150.0.
        num_atoms (int, optional):  hyperparameters for the distributional critic in
            mad4pg. Defaults to 51.
        seed (int, optional): random seed for network initialization.

    Returns:
        Mapping[str, types.TensorTransformation]: returned agent networks.
    """
    return make_default_networks_maddpg(
        environment_spec=environment_spec,
        agent_net_keys=agent_net_keys,
        policy_networks_layer_sizes=policy_networks_layer_sizes,
        critic_networks_layer_sizes=critic_networks_layer_sizes,
        sigma=sigma,
        archecture_type=archecture_type,
        vmin=vmin,
        vmax=vmax,
        num_atoms=num_atoms,
        seed=seed,
    )
