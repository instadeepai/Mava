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

"""Jax IPPO system networks."""
import dataclasses
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import chex
import haiku as hk
import jax
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
from dm_env import specs as dm_specs

from mava import specs as mava_specs
from mava.systems.idqn.idqn_network import IDQNNetwork
from mava.utils.jax_training_utils import action_mask_categorical_policies

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]


def make_discrete_network(
    environment_spec: specs.EnvironmentSpec,
    base_key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
) -> IDQNNetwork:
    """Create DQN network for environments with discrete action spaces.

    Args:
        environment_spec: environment spec
        key: jax random key for initializing network parameters
        policy_layer_sizes: sizes of hidden layers for the policy network
        critic_layer_sizes: sizes of hidden layers for the critic network
        policy_recurrent_layer_sizes: Optionally add recurrent layers to the policy
        recurrent_architecture_fn: Architecture to use for the recurrent units,
        e.g. LSTM or GRU.
        policy_layers_after_recurrent: sizes of hidden layers for the
            policy network after the recurrent layers. This is only used if a
            recurrent architecture is used.
        orthogonal_initialisation: Whether network weights should be
            initialised orthogonally.
        activation_function: activation function to be used for
            network hidden layers.
        policy_network_head_weight_gain: value for scaling the policy
            network final layer weights by.

    Returns:
        PPONetworks class
    """

    num_actions = environment_spec.actions.num_values

    @hk.without_apply_rng
    @hk.transform
    def policy_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        """Create a policy network function and transform it using Haiku.

        Args:
            inputs: The inputs required for hk.DeepRNN or hk.Sequential.

        Returns:
            FeedForwardNetwork class
        """
        # Add the observation network and an MLP network.
        policy_network = [
            hk.nets.MLP(
                (*policy_layer_sizes, num_actions),
                activation=activation_function,
            ),
        ]

        return hk.Sequential(policy_network)(inputs)

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    base_key, network_key = jax.random.split(base_key)

    policy_params = policy_fn.init(network_key, dummy_obs)  # type: ignore

    base_key, network_key = jax.random.split(base_key)

    # Create PPONetworks to add functionality required by the agent.
    return IDQNNetwork(
        network=policy_fn,
        policy_params=policy_params,
    )


def make_networks(
    spec: specs.EnvironmentSpec,
    base_key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
) -> IDQNNetwork:
    """Function for creating PPO networks to be used.

    These networks will be different depending on whether the
    environment has a discrete or continuous action space.

    Args:
        spec: specifications of training environment
        base_key: pseudo-random value used to initialise distributions
        policy_layer_sizes: size of each layer of the policy network
        critic_layer_sizes: size of each layer of the critic network
        policy_recurrent_layer_sizes: Optionally add recurrent layers to the policy
        recurrent_architecture_fn: Architecture to use for the recurrent units,
            e.g. LSTM or GRU.
        policy_layers_after_recurrent: sizes of hidden layers for the policy
        network after the recurrent layers. This is only used if a
        recurrent architecture is used.
        orthogonal_initialisation: Whether network weights should be
            initialised orthogonally.
        activation_function: activation function to be used for
            network hidden layers.
        policy_network_head_weight_gain: value for scaling the policy
            network final layer weights by.

    Returns:
        make_discrete_networks: function to create a discrete network

    Raises:
        NotImplementedError: Raises an error if continuous network is not
                        available
    """
    if isinstance(spec.actions, specs.DiscreteArray):
        return make_discrete_network(
            environment_spec=spec,
            base_key=base_key,
            policy_layer_sizes=policy_layer_sizes,
        )

    else:
        raise NotImplementedError(
            "Continuous networks not implemented yet."
            + "See: https://github.com/deepmind/acme/blob/"
            + "master/acme/agents/jax/ppo/networks.py"
        )


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    base_key: List[int],
    net_spec_keys: Dict[str, str] = {},
    policy_layer_sizes: Sequence[int] = (
        64,
        64,
    ),
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
) -> Dict[str, Any]:
    """Create default PPO networks

    Args:
        environment_spec: mava multi-agent environment spec
        agent_net_keys: dictionary specifying which networks are
            used by which agent
        base_key: jax random key to be used for network initialization
        net_spec_keys: keys for each agent network
        policy_layer_sizes: policy network layers
        critic_layer_sizes: critic network layers
        policy_recurrent_layer_sizes: Optionally add recurrent layers to the policy
        recurrent_architecture_fn: Architecture to use for the recurrent units,
            e.g. LSTM or GRU.
        policy_layers_after_recurrent: sizes of hidden layers for the policy
            network after the recurrent layers. This is only used if a
            recurrent architecture is used.
        orthogonal_initialisation: whether network weights should be
            initialised orthogonally. This will initialise all hidden
            layers weights with scale sqrt(2) and all hidden layer
            biases with a constant value of 0.0. The policy network
            output head weights are orthogonally initialised with scale 0.01 and
            the critic network output head weights are orthogonally initialised
            with scale 1.0.Scale value obtained from:
            https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/ # noqa: E501
        activation_function: activation function to be used for
            network hidden layers.
        policy_network_head_weight_gain: value for scaling the policy
            network final layer weights by.

    Returns:
        networks: networks created to given spec
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
        networks[net_key] = make_networks(
            specs[net_key],
            base_key=base_key,
            policy_layer_sizes=policy_layer_sizes,
            activation_function=activation_function,
        )

    # No longer returning a dictionary since this is handled in PPONetworks above
    return networks
