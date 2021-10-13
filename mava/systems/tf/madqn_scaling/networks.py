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

import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf.networks.atari import DQNAtariNetwork, AtariTorso

from mava import specs as mava_specs
from mava.components.tf import networks
from mava.components.tf.networks import epsilon_greedy_action_selector
from mava.utils.enums import ArchitectureType, Network


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    net_spec_keys: Dict[str, str] = {},
    q_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = None,
    archecture_type: ArchitectureType = ArchitectureType.feedforward,
    network_type: Network = Network.mlp,
    seed: Optional[int] = None,
) -> Mapping[str, types.TensorTransformation]:
    """Default networks for madqn.

    Args:
        environment_spec (mava_specs.MAEnvironmentSpec): description of the action and
            observation spaces etc. for each agent in the system.
        agent_net_keys: (dict, optional): specifies what network each agent uses.
            Defaults to {}.
        q_networks_layer_sizes (Union[Dict[str, Sequence], Sequence], optional):
            size of policy networks.
        archecture_type (ArchitectureType, optional): archecture used
            for agent networks. Can be feedforward or recurrent.
            Defaults to ArchitectureType.feedforward.
        network_type (Network, optional): Agent network type.
            Can be mlp, atari_dqn_network or coms_network.
            Defaults to Network.mlp.
        seed (int, optional): random seed for network initialization.

    Returns:
        Mapping[str, types.TensorTransformation]: returned agent networks.
    """

    # Set Policy function and layer size.
    # Default size per arch type.
    if archecture_type == ArchitectureType.feedforward:
        if not q_networks_layer_sizes:
            q_networks_layer_sizes = (512, 512, 256)
        q_network_func = snt.Sequential
    elif archecture_type == ArchitectureType.recurrent:
        if not q_networks_layer_sizes:
            q_networks_layer_sizes = (128, 128)
        q_network_func = snt.DeepRNN

    assert q_networks_layer_sizes is not None
    assert q_network_func is not None

    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    if isinstance(q_networks_layer_sizes, Sequence):
        q_networks_layer_sizes = {
            key: q_networks_layer_sizes for key in specs.keys()
        }

    def action_selector_fn(
        q_values: types.NestedTensor,
        legal_actions: types.NestedTensor,
        epsilon: Optional[tf.Variable] = None,
    ) -> types.NestedTensor:
        return epsilon_greedy_action_selector(
            action_values=q_values, legal_actions_mask=legal_actions, epsilon=epsilon
        )

    q_networks = {}
    action_selectors = {}
    observation_networks = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = specs[key].actions.num_values

        # Create the policy network.
        if network_type == Network.atari_dqn_network:
            observation_network = AtariTorso()
        else:
            observation_network = snt.Linear(100) # Make this the identity transformation tf2_utils.to_sonnet_module(v)

        if archecture_type == ArchitectureType.feedforward:
            q_network = [
                networks.LayerNormMLP(
                    list(q_networks_layer_sizes[key]) + [num_dimensions],
                    activate_final=False,
                    seed=seed,
                ),
            ]
        elif archecture_type == ArchitectureType.recurrent:
            q_network = [
                networks.LayerNormMLP(
                    q_networks_layer_sizes[key][:-1],
                    activate_final=True,
                    seed=seed,
                ),
                snt.LSTM(q_networks_layer_sizes[key][-1]),
                snt.Linear(num_dimensions),
            ]

        q_network = q_network_func(q_network)

        # Store Networks
        q_networks[key] = q_network
        action_selectors[key] = action_selector_fn
        observation_networks[key] = observation_network

    return {
        "observation_networks": observation_networks,
        "q_networks": q_networks,
        "action_selectors": action_selectors,
    }
