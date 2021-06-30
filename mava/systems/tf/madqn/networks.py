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
from acme.tf import networks
from acme.tf.networks.atari import DQNAtariNetwork

from mava import specs as mava_specs
from mava.components.tf.networks import epsilon_greedy_action_selector
from mava.utils.enums import ArchitectureType, Network


# Default networks for madqn
# TODO Use fingerprints variable
def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    shared_weights: bool = True,
    archecture_type: ArchitectureType = ArchitectureType.feedforward,
    network_type: Network = Network.mlp,
    fingerprints: bool = False,
) -> Mapping[str, types.TensorTransformation]:

    # Set Policy function and layer size
    if archecture_type == ArchitectureType.feedforward:
        policy_network_func = snt.Sequential
    elif archecture_type == ArchitectureType.recurrent:
        policy_networks_layer_sizes = (128, 128)
        policy_network_func = snt.DeepRNN

    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    if isinstance(policy_networks_layer_sizes, Sequence):
        policy_networks_layer_sizes = {
            key: policy_networks_layer_sizes for key in specs.keys()
        }

    def action_selector_fn(
        q_values: types.NestedTensor,
        legal_actions: types.NestedTensor,
        epsilon: Optional[tf.Variable] = None,
    ) -> types.NestedTensor:
        return epsilon_greedy_action_selector(
            action_values=q_values, legal_actions_mask=legal_actions, epsilon=epsilon
        )

    policy_networks = {}
    action_selectors = {}
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = specs[key].actions.num_values

        # Create the policy network.
        if network_type == Network.atari_dqn_network:
            policy_network = DQNAtariNetwork(num_dimensions)
        else:
            if archecture_type == ArchitectureType.feedforward:
                policy_network = [
                    networks.LayerNormMLP(
                        list(policy_networks_layer_sizes[key]) + [num_dimensions],
                        activate_final=False,
                    ),
                ]
            elif archecture_type == ArchitectureType.recurrent:
                policy_network = [
                    networks.LayerNormMLP(
                        policy_networks_layer_sizes[key][:-1], activate_final=True
                    ),
                    snt.LSTM(policy_networks_layer_sizes[key][-1]),
                    snt.Linear(num_dimensions),
                ]

            policy_network = policy_network_func(policy_network)

        # epsilon greedy action selector
        policy_networks[key] = policy_network
        action_selectors[key] = action_selector_fn

    return {
        "q_networks": policy_networks,
        "action_selectors": action_selectors,
    }
