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
from typing import Dict, Optional, Sequence, Union

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
from acme.tf import utils as tf2_utils
from dm_env import specs

from mava import specs as mava_specs
from mava.components.tf import networks
from mava.utils.enums import ArchitectureType

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray


# TODO Update for recurrent version.
def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    net_spec_keys: Dict[str, str] = {},
    policy_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (
        256,
        256,
        256,
    ),
    critic_networks_layer_sizes: Union[Dict[str, Sequence], Sequence] = (512, 512, 256),
    archecture_type: ArchitectureType = ArchitectureType.feedforward,
    seed: Optional[int] = None,
) -> Dict[str, snt.Module]:
    """Default networks for mappo.

    Args:
        environment_spec (mava_specs.MAEnvironmentSpec): description of the action and
            observation spaces etc. for each agent in the system.
        agent_net_keys: (dict, optional): specifies what network each agent uses.
            Defaults to {}.
        net_spec_keys: specifies the specs of each network.
        policy_networks_layer_sizes (Union[Dict[str, Sequence], Sequence], optional):
            size of policy networks. Defaults to (256, 256, 256).
        critic_networks_layer_sizes (Union[Dict[str, Sequence], Sequence], optional):
            size of critic networks. Defaults to (512, 512, 256).
        seed (int, optional): random seed for network initialization.

    Raises:
        ValueError: Unknown action_spec type, if actions aren't DiscreteArray
            or BoundedArray.

    Returns:
        Dict[str, snt.Module]: returned agent networks.
    """

    # Create agent_type specs.
    specs = environment_spec.get_agent_specs()
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    # Set Policy function and layer size
    # Default size per arch type.
    if archecture_type == ArchitectureType.feedforward:
        if not policy_networks_layer_sizes:
            policy_networks_layer_sizes = (
                256,
                256,
                256,
            )
        policy_network_func = snt.Sequential
    elif archecture_type == ArchitectureType.recurrent:
        if not policy_networks_layer_sizes:
            policy_networks_layer_sizes = (128, 128)
        policy_network_func = snt.DeepRNN

    if isinstance(policy_networks_layer_sizes, Sequence):
        policy_networks_layer_sizes = {
            key: policy_networks_layer_sizes for key in specs.keys()
        }
    if isinstance(critic_networks_layer_sizes, Sequence):
        critic_networks_layer_sizes = {
            key: critic_networks_layer_sizes for key in specs.keys()
        }

    observation_networks = {}
    policy_networks = {}
    critic_networks = {}
    for key in specs.keys():

        # Create the shared observation network; here simply a state-less operation.
        observation_network = tf2_utils.to_sonnet_module(tf.identity)

        # Note: The discrete case must be placed first as it inherits from BoundedArray.
        if isinstance(specs[key].actions, dm_env.specs.DiscreteArray):  # discrete
            num_actions = specs[key].actions.num_values

            if archecture_type == ArchitectureType.feedforward:
                policy_layers = [
                    networks.LayerNormMLP(
                        tuple(policy_networks_layer_sizes[key]) + (num_actions,),
                        activate_final=False,
                        seed=seed,
                    ),
                ]
            elif archecture_type == ArchitectureType.recurrent:
                policy_layers = [
                    networks.LayerNormMLP(
                        policy_networks_layer_sizes[key][:-1],
                        activate_final=True,
                        seed=seed,
                    ),
                    snt.LSTM(policy_networks_layer_sizes[key][-1]),
                    networks.LayerNormMLP(
                        (num_actions,),
                        activate_final=False,
                        seed=seed,
                    ),
                ]

            policy_network = policy_network_func(policy_layers)

        elif isinstance(specs[key].actions, dm_env.specs.BoundedArray):  # continuous
            num_actions = np.prod(specs[key].actions.shape, dtype=int)
            # No recurrent setting implemented yet.
            assert archecture_type == ArchitectureType.feedforward
            policy_network = snt.Sequential(
                [
                    networks.LayerNormMLP(
                        policy_networks_layer_sizes[key], activate_final=True, seed=seed
                    ),
                    networks.MultivariateNormalDiagHead(
                        num_dimensions=num_actions,
                        w_init=tf.initializers.VarianceScaling(1e-4, seed=seed),
                        b_init=tf.initializers.Zeros(),
                    ),
                    networks.TanhToSpec(specs[key].actions),
                ]
            )
        else:
            raise ValueError(f"Unknown action_spec type, got {specs[key].actions}.")

        critic_network = snt.Sequential(
            [
                networks.LayerNormMLP(
                    list(critic_networks_layer_sizes[key]) + [1],
                    activate_final=False,
                    seed=seed,
                ),
            ]
        )

        observation_networks[key] = observation_network
        policy_networks[key] = policy_network
        critic_networks[key] = critic_network
    return {
        "policies": policy_networks,
        "critics": critic_networks,
        "observations": observation_networks,
    }
