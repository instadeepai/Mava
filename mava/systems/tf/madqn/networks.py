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

"""Default networks for MADQN systems."""
from typing import Dict, Mapping, Optional, Sequence, Union

import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf.networks import AtariTorso
from dm_env import specs

from mava import specs as mava_specs
from mava.components.tf import networks
from mava.components.tf.networks.epsilon_greedy import EpsilonGreedy
from mava.utils.enums import ArchitectureType

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    net_spec_keys: Dict[str, str] = {},
    value_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = None,
    architecture_type: ArchitectureType = ArchitectureType.feedforward,
    atari_torso_observation_network: bool = False,
    seed: Optional[int] = None,
) -> Mapping[str, types.TensorTransformation]:
    """Default networks for madqn.

    Args:
        environment_spec: description of the action and
            observation spaces etc. for each agent in the system.
        agent_net_keys: specifies what network each agent uses.
        net_spec_keys: specifies the specs of each network.
        value_networks_layer_sizes: size of value networks.
        architecture_type: architecture used
            for agent networks. Can be feedforward or recurrent.
            Defaults to ArchitectureType.feedforward.
        seed: random seed for network initialization.

    Returns:
        returned agent networks.
    """
    # Set Policy function and layer size
    # Default size per arch type.
    if architecture_type == ArchitectureType.feedforward:
        if not value_networks_layer_sizes:
            value_networks_layer_sizes = (
                256,
                256,
                256,
            )
        value_network_func = snt.Sequential
    elif architecture_type == ArchitectureType.recurrent:
        if not value_networks_layer_sizes:
            value_networks_layer_sizes = (64, 64)
        value_network_func = snt.DeepRNN

    assert value_networks_layer_sizes is not None
    assert value_network_func is not None

    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    if isinstance(value_networks_layer_sizes, Sequence):
        value_networks_layer_sizes = {
            key: value_networks_layer_sizes for key in specs.keys()
        }

    observation_networks = {}
    value_networks = {}
    action_selectors = {}
    for key, spec in specs.items():
        num_actions = spec.actions.num_values

        # An optional network to process observations
        if not atari_torso_observation_network:
            observation_network = tf2_utils.to_sonnet_module(tf.identity)
        else:
            observation_network = AtariTorso()

        # Create the policy network.
        if architecture_type == ArchitectureType.feedforward:
            value_network = [
                networks.LayerNormMLP(
                    value_networks_layer_sizes[key], activate_final=True, seed=seed
                ),
            ]
        elif architecture_type == ArchitectureType.recurrent:
            value_network = [
                networks.LayerNormMLP(
                    value_networks_layer_sizes[key][:-1],
                    activate_final=True,
                    seed=seed,
                ),
                snt.GRU(value_networks_layer_sizes[key][-1]),
            ]

        value_network += [
            networks.NearZeroInitializedLinear(num_actions, seed=seed),
        ]

        value_network = value_network_func(value_network)

        observation_networks[key] = observation_network
        value_networks[key] = value_network
        action_selectors[key] = EpsilonGreedy

    return {
        "values": value_networks,
        "action_selectors": action_selectors,
        "observations": observation_networks,
    }
