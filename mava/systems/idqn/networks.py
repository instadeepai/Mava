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

"""Jax IDQN system networks."""
from typing import Any, Callable, Dict, Optional, Sequence

import haiku as hk
import jax
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils

from mava import specs as mava_specs
from mava.systems.idqn.idqn_network import IDQNNetwork


def make_network(
    environment_spec: specs.EnvironmentSpec,
    base_key: jax.random.KeyArray,
    policy_layer_sizes: Sequence[int],
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    observation_network: Optional[Callable] = None,
) -> IDQNNetwork:
    """Create DQN network for environments with discrete action spaces.

    Args:
        environment_spec: environment spec
        base_key: jax random key for initializing network parameters
        policy_layer_sizes: sizes of hidden layers for the policy network
        activation_function: activation function to be used for
            network hidden layers.
        observation_network: optional network for processing observations.
            Defaults to nothing

    Returns:
        A single IDQN network
    """

    num_actions = environment_spec.actions.num_values

    @hk.without_apply_rng
    @hk.transform
    def policy_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        """Create a policy network function and transform it using Haiku.

        Args:
            inputs: The inputs required for hk.Sequential.

        Returns:
            FeedForwardNetwork class
        """
        # Add the observation network and an MLP network.
        q_value_network = []
        if observation_network is not None:
            q_value_network.append(observation_network)

        q_value_network.append(
            hk.nets.MLP(
                (*policy_layer_sizes, num_actions),
                activation=activation_function,
            ),
        )

        return hk.Sequential(q_value_network)(inputs)

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    base_key, network_key = jax.random.split(base_key)

    policy_params = policy_fn.init(network_key, dummy_obs)

    base_key, network_key = jax.random.split(base_key)

    return IDQNNetwork(
        network=policy_fn,
        policy_params=policy_params,
    )


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    base_key: jax.random.KeyArray,
    net_spec_keys: Dict[str, str] = {},
    policy_layer_sizes: Sequence[int] = (
        64,
        64,
    ),
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    observation_network: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Create default IDQN networks (one per agent)

    Args:
        environment_spec: mava multi-agent environment spec
        agent_net_keys: dictionary specifying which networks are
            used by which agent
        base_key: jax random key to be used for network initialization
        net_spec_keys: keys for each agent network
        policy_layer_sizes: policy network layers
        activation_function: activation function to be used for network hidden layers.
        observation_network: optional network for processing observations

    Returns:
        networks: IDQN networks created to given spec
    """

    # Create agent_type specs.
    specs = environment_spec.get_agent_environment_specs()
    if not net_spec_keys:
        specs = {
            agent_net_keys[agent_key]: specs[agent_key] for agent_key in specs.keys()
        }
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_network(
            specs[net_key],
            base_key=base_key,
            policy_layer_sizes=policy_layer_sizes,
            activation_function=activation_function,
            observation_network=observation_network,
        )

    return networks
