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
from typing import Mapping, Optional

import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import networks

from mava import specs as mava_specs
from mava.components.tf.networks import epsilon_greedy_action_selector
from mava.components.tf.networks.communication import CommunicationNetwork
from mava.utils.enums import ArchitectureType


# Default networks for Dial
def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    message_size: int = 1,
    shared_weights: bool = True,
    archecture_type: ArchitectureType = ArchitectureType.recurrent,
) -> Mapping[str, types.TensorTransformation]:
    """Creates networks used by the agents."""

    assert (
        archecture_type == ArchitectureType.recurrent
    ), "Dial currently supports recurrent architectures."

    specs = environment_spec.get_agent_specs()

    # Create agent_type specs
    if shared_weights:
        type_specs = {key.split("_")[0]: specs[key] for key in specs.keys()}
        specs = type_specs

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

        q_network = CommunicationNetwork(
            networks.LayerNormMLP(
                (128,),
                activate_final=True,
            ),
            networks.LayerNormMLP(
                (128,),
                activate_final=True,
            ),
            snt.GRU(128),
            snt.Sequential(
                [
                    networks.LayerNormMLP((128,), activate_final=True),
                    networks.NearZeroInitializedLinear(num_dimensions),
                    networks.TanhToSpec(specs[key].actions),
                ]
            ),
            snt.Sequential(
                [
                    networks.LayerNormMLP((128, message_size), activate_final=True),
                ]
            ),
            message_size=message_size,
        )

        # epsilon greedy action selector
        action_selector = action_selector_fn

        q_networks[key] = q_network
        action_selectors[key] = action_selector

    return {
        "q_networks": q_networks,
        "action_selectors": action_selectors,
    }
