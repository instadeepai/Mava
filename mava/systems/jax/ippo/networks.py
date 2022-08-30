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
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import chex
import haiku as hk  # type: ignore
import jax
import jax.numpy as jnp
import numpy as np
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
from dm_env import specs as dm_specs
from jax import jit

from mava import specs as mava_specs
from mava.utils.jax_training_utils import action_mask_categorical_policies

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]


@dataclasses.dataclass
class PPONetworks:
    """Class to implement the networks for the PPO algorithm"""

    def __init__(
        self,
        network: networks_lib.FeedForwardNetwork,
        params: networks_lib.Params,
        log_prob: Optional[networks_lib.LogProbFn] = None,
        entropy: Optional[EntropyFn] = None,
        sample: Optional[networks_lib.SampleFn] = None,
    ) -> None:
        """Initialises the PPO network Class.

        Args:
            network: neural network representing the agent policy function.
            params: values parameterising the network.
            log_prob: function used to calculate the log prob of an agent's action.
            entropy: function used to calculate the entropy of the agent policy.
            sample: function used to select an action from the policy.
        """
        self.network = network
        self.params = params
        self.log_prob = log_prob
        self.entropy = entropy
        self.sample = sample

        @jit
        def forward_fn(
            params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
            key: networks_lib.PRNGKey,
            mask: chex.Array = None,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Forward function for PPO network

            Args:
                params: values parameterising the network.
                observations: agent observations
                key: pseudo-random value used to initialise distributions
                mask: action mask which removes illegal actions

            Returns:
                actions: agent action
                log_prob: log prob of the chosen action
            """
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            distribution, _ = self.network.apply(params, observations)
            if mask is not None:
                distribution = action_mask_categorical_policies(distribution, mask)

            actions = jax.numpy.squeeze(distribution.sample(seed=key))
            log_prob = distribution.log_prob(actions)

            return actions, log_prob

        self.forward_fn = forward_fn

    def get_action(
        self,
        observations: networks_lib.Observation,
        key: networks_lib.PRNGKey,
        mask: chex.Array = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Gets an action from the network from given observation

        Args:
           observations: agent observations
           key: pseudo-random value used to initialise distributions
           mask: action mask which removes illegal actions

        Returns:
            actions: agent action
            log_prob: log prob of the chosen action

        """
        actions, log_prob = self.forward_fn(self.params, observations, key, mask)
        actions = np.array(actions, dtype=np.int64)
        log_prob = np.squeeze(np.array(log_prob, dtype=np.float32))
        return actions, {"log_prob": log_prob}

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """Gets value of observation

        Args:
            observations: agent observations

        Returns:
            value: estimated value of observation

        """
        _, value = self.network.apply(self.params, observations)
        return value


# This class is made to replicate the behaviour of the categorical value head
# which squeezes the value inside the __call__ method before returning it.
# please see
# https://github.com/deepmind/acme/blob/70e1e6b694d79d94f1bed13d55cda5c1837a10f3/acme/jax/networks/distributional.py#L284 # noqa: E501
class ValueHead(hk.Module):
    """Network head that produces a value."""

    def __init__(
        self,
        name: Optional[str] = None,
    ):
        """Initialize the class"""
        super().__init__(name=name)
        self._value_layer = hk.Linear(1)

    def __call__(self, inputs: jnp.ndarray) -> Any:
        """Return output given network inputs."""
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return value


@dataclasses.dataclass
class PPOSeparateNetworks:
    """Separate policy and critic networks for IPPO."""

    def __init__(
        self,
        policy_network: networks_lib.FeedForwardNetwork,
        policy_params: networks_lib.Params,
        critic_network: networks_lib.FeedForwardNetwork,
        critic_params: networks_lib.Params,
        log_prob: Optional[networks_lib.LogProbFn] = None,
        entropy: Optional[EntropyFn] = None,
        sample: Optional[networks_lib.SampleFn] = None,
    ) -> None:
        """Initiliaze networks class

        Args:
            policy_network: network to be used by the policy
            policy_params: parameters for the policy network
            critic_network: network to be used by the critic
            critic_params: parameters for the critic network
            log_prob: lambda function for getting the log probability of
                a particular action from a distribution.
                Defaults to None.
            entropy: lambda function for getting the entropy of
                a particular distribution.
                Defaults to None.
            sample: lambda function for sampling an action
                from a distribution.
                Defaults to None.
        """
        self.policy_network = policy_network
        self.policy_params = policy_params
        self.critic_network = critic_network
        self.critic_params = critic_params
        self.log_prob = log_prob
        self.entropy = entropy
        self.sample = sample

        @jit
        def forward_fn(
            policy_params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
            key: networks_lib.PRNGKey,
            mask: chex.Array = None,
        ) -> Tuple[jnp.ndarray, jnp.ndarray]:
            """Get actions and relevant log probabilities from the \
                policy network given some observations.

            Args:
                policy_params: parameters of the policy network
                observations: agent observations
                key: jax random key for sampling from the policy
                    distribution
                mask: optional mask for selecting only legal actions
                    Defaults to None.

            Returns:
                Tuple (actions, log_prob): sampled actions and relevant
                    log probabilities of sampled actions
            """
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            distribution = self.policy_network.apply(policy_params, observations)
            if mask is not None:
                distribution = action_mask_categorical_policies(distribution, mask)

            actions = jax.numpy.squeeze(distribution.sample(seed=key))
            log_prob = distribution.log_prob(actions)

            return actions, log_prob

        self.forward_fn = forward_fn

    def get_action(
        self,
        observations: networks_lib.Observation,
        key: networks_lib.PRNGKey,
        mask: chex.Array = None,
    ) -> Tuple[np.ndarray, Dict]:
        """Get actions from policy network given observations."""
        actions, log_prob = self.forward_fn(self.policy_params, observations, key, mask)
        actions = np.array(actions, dtype=np.int64)
        log_prob = np.squeeze(np.array(log_prob, dtype=np.float32))
        return actions, {"log_prob": log_prob}

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """Get state value from critic network given observations."""
        value = self.critic_network.apply(self.critic_params, observations)
        return value


def make_ppo_network(
    network: networks_lib.FeedForwardNetwork, params: Dict[str, jnp.ndarray]
) -> PPONetworks:
    """Instantiate PPO network class which has shared hidden layers with unique\
        policy and value heads.

    Args:
        network: feedforward network representing the agent policy function
        params: values parameterising the network.

    Returns:
        PPONetworks: PPO network class
    """
    return PPONetworks(
        network=network,
        params=params,
        log_prob=lambda distribution, action: distribution.log_prob(action),
        entropy=lambda distribution: distribution.entropy(),
        sample=lambda distribution, key: distribution.sample(seed=key),
    )


def make_ppo_networks(
    policy_network: networks_lib.FeedForwardNetwork,
    policy_params: Dict[str, jnp.ndarray],
    critic_network: networks_lib.FeedForwardNetwork,
    critic_params: Dict[str, jnp.ndarray],
) -> PPOSeparateNetworks:
    """Instantiate PPO networks class which has separate policy and \
        critic networks."""
    return PPOSeparateNetworks(
        policy_network=policy_network,
        policy_params=policy_params,
        critic_network=critic_network,
        critic_params=critic_params,
        log_prob=lambda distribution, action: distribution.log_prob(action),
        entropy=lambda distribution: distribution.entropy(),
        sample=lambda distribution, key: distribution.sample(seed=key),
    )


def make_networks(
    spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int] = (
        256,
        256,
        256,
    ),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    observation_network: Callable = utils.batch_concat,
    single_network: bool = True,
) -> Union[PPONetworks, PPOSeparateNetworks]:
    """Function for creating PPO networks to be used.

    These networks will be different depending on whether the
    environment has a discrete or continuous action space.

    Args:
        spec: specifications of training environment
        key: pseudo-random value used to initialise distributions
        policy_layer_sizes: size of each layer of the policy network
        critic_layer_sizes: size of each layer of the critic network
        observation_network: Network used for feature extraction layers
        single_network: If a shared represnetation netowrk should be used.

    Returns:
        make_discrete_networks: function to create a discrete network
        make_continuous_networks: function to create a continuous network

    Raises:
        NotImplementedError: Raises an error if continous network is not
                        available
    """
    if isinstance(spec.actions, specs.DiscreteArray):
        return make_discrete_networks(
            environment_spec=spec,
            key=key,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            observation_network=observation_network,
            single_network=single_network,
        )

    else:
        raise NotImplementedError(
            "Continuous networks not implemented yet."
            + "See: https://github.com/deepmind/acme/blob/"
            + "master/acme/agents/jax/ppo/networks.py"
        )


def make_discrete_networks(
    environment_spec: specs.EnvironmentSpec,
    key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    critic_layer_sizes: Sequence[int],
    observation_network: Callable = utils.batch_concat,
    single_network: bool = True,
    # default behaviour is to flatten observations
) -> Union[PPONetworks, PPOSeparateNetworks]:
    """Create PPO network for environments with discrete action spaces.

    Args:
        environment_spec: environment spec
        key: jax random key for initializing network parameters
        policy_layer_sizes: sizes of hidden layers for the policy network
        critic_layer_sizes: sizes of hidden layers for the critic network
        observation_network: optional network for processing observations.
            Defaults to utils.batch_concat.
        single_network: True if shared layer network with separate heads should be used
            for policy and critic and False if separate policy and critic networks
            should be used.
            Defaults to True.

    Returns:
        PPONetworks class
    """

    num_actions = environment_spec.actions.num_values

    # TODO (dries): Investigate if one forward_fn function is slower
    # than having a policy_fn and critic_fn. Maybe jit solves
    # this issue. Having one function makes obs network calculations
    # easier.

    if single_network:

        def forward_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
            policy_value_network = hk.Sequential(
                [
                    observation_network,
                    hk.nets.MLP(policy_layer_sizes, activation=jax.nn.relu),
                    networks_lib.CategoricalValueHead(num_values=num_actions),
                ]
            )
            return policy_value_network(inputs)

        # Transform into pure functions.
        forward_fn = hk.without_apply_rng(hk.transform(forward_fn))

        dummy_obs = utils.zeros_like(environment_spec.observations.observation)
        dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

        network_key, key = jax.random.split(key)
        params = forward_fn.init(network_key, dummy_obs)  # type: ignore

        # Create PPONetworks to add functionality required by the agent.
        return make_ppo_network(network=forward_fn, params=params)

    else:

        def policy_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
            policy_network = hk.Sequential(
                [
                    observation_network,
                    hk.nets.MLP(policy_layer_sizes, activation=jax.nn.relu),
                    networks_lib.CategoricalHead(num_values=num_actions),
                ]
            )
            return policy_network(inputs)

        def critic_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
            critic_network = hk.Sequential(
                [
                    observation_network,
                    hk.nets.MLP(critic_layer_sizes, activation=jax.nn.relu),
                    ValueHead(),
                ]
            )
            return critic_network(inputs)

        # Transform into pure functions.
        policy_fn = hk.without_apply_rng(hk.transform(policy_fn))
        critic_fn = hk.without_apply_rng(hk.transform(critic_fn))

        dummy_obs = utils.zeros_like(environment_spec.observations.observation)
        dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

        network_key, key = jax.random.split(key)
        policy_params = policy_fn.init(network_key, dummy_obs)  # type: ignore

        network_key, key = jax.random.split(key)
        critic_params = critic_fn.init(network_key, dummy_obs)  # type: ignore

        # Create PPONetworks to add functionality required by the agent.
        return make_ppo_networks(
            policy_network=policy_fn,
            policy_params=policy_params,
            critic_network=critic_fn,
            critic_params=critic_params,
        )


def make_default_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    rng_key: List[int],
    net_spec_keys: Dict[str, str] = {},
    policy_layer_sizes: Sequence[int] = (
        256,
        256,
        256,
    ),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    observation_network: Callable = utils.batch_concat,
    single_network: bool = True,
) -> Dict[str, Any]:
    """Create default PPO networks

    Args:
        environment_spec: mava multi-agent environment spec
        agent_net_keys: dictionary specifiying which networks are
                        used by which agent
        rng_key: jax random key to be used for network initialization
        net_spec_keys: keys for each agent network
        policy_layer_sizes: policy network layers
        critic_layer_sizes: critic network layers
        observation_network: network for processing environment observations
                             defaults to flattening observations but could be
                             a CNN or similar observation processing network
        single_network: details whether a single network with a policy and value
                        head should be used if true or whether separate policy
                        and critic networks should be used if false.

    Returns:
        networks: networks created to given spec
    """

    # Create agent_type specs.
    specs = environment_spec.get_agent_environment_specs()
    if not net_spec_keys:
        specs = {agent_net_keys[key]: specs[key] for key in specs.keys()}
    else:
        specs = {net_key: specs[value] for net_key, value in net_spec_keys.items()}

    networks: Dict[str, Any] = {}
    for net_key in specs.keys():
        networks[net_key] = make_networks(
            specs[net_key],
            key=rng_key,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            observation_network=observation_network,
            single_network=single_network,
        )

    return {
        "networks": networks,
    }
