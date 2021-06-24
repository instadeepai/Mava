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

valid_dqn_network_types = ["mlp", "atari"]


# Default networks for madqn
# TODO Use fingerprints variable
def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    q_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        512,
        256,
    ),
    shared_weights: bool = True,
    network_type: str = "mlp",
    fingerprints: bool = False,
) -> Mapping[str, types.TensorTransformation]:

    assert (
        network_type.lower() in valid_dqn_network_types
    ), f"Invalid network_type, valid options are {valid_dqn_network_types}"
    """Creates networks used by the agents."""

    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

    if isinstance(q_networks_layer_sizes, Sequence):
        q_networks_layer_sizes = {key: q_networks_layer_sizes for key in specs.keys()}

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
    for key in specs.keys():

        # Get total number of action dimensions from action spec.
        num_dimensions = specs[key].actions.num_values

        # Create the policy network.
        if network_type.lower() == "mlp":
            q_network = snt.Sequential(
                [
                    networks.LayerNormMLP(
                        q_networks_layer_sizes[key], activate_final=True
                    ),
                    networks.NearZeroInitializedLinear(num_dimensions),
                ]
            )
        elif network_type.lower() == "atari":
            q_network = DQNAtariNetwork(num_dimensions)

        # epsilon greedy action selector
        action_selector = action_selector_fn

        q_networks[key] = q_network
        action_selectors[key] = action_selector

    return {
        "q_networks": q_networks,
        "action_selectors": action_selectors,
    }
