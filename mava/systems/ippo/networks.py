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
import haiku as hk  # type: ignore
import jax
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
from dm_env import specs as dm_specs

from mava import specs as mava_specs
from mava.utils.jax_training_utils import action_mask_categorical_policies

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]


# TODO JAX Networks should be stateless.
@dataclasses.dataclass
class PPONetworks:
    """Separate policy and critic networks for IPPO."""

    def __init__(
        self,
        policy_network: networks_lib.FeedForwardNetwork,
        policy_params: networks_lib.Params,
        policy_init_state: Any,
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
        self.policy_init_state = policy_init_state
        self.critic_network = critic_network
        self.critic_params = critic_params
        self.log_prob = log_prob
        self.entropy = entropy
        self.sample = sample

        def forward_fn(
            policy_params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
            key: networks_lib.PRNGKey,
            mask: chex.Array = None,
            policy_state: Tuple[jnp.ndarray] = None,
        ) -> Tuple[Any, Any, Any]:
            """Get actions and relevant log probabilities from the \
                policy network given some observations.

            Args:
                policy_params: parameters of the policy network
                observations: agent observations
                key: jax random key for sampling from the policy
                    distribution
                policy_state: Optional state used for recurrent policies
                mask: optional mask for selecting only legal actions
                    Defaults to None.

            Returns:
                Tuple (actions, log_prob): sampled actions and relevant
                    log probabilities of sampled actions
            """
            # The parameters of the network might change. So it has to
            # be fed into the jitted function.
            if not self.policy_init_state:
                distribution = self.policy_network.apply(policy_params, observations)
            else:
                distribution, policy_state = self.policy_network.apply(
                    policy_params, [observations, policy_state]
                )

            if mask is not None:
                distribution = action_mask_categorical_policies(distribution, mask)

            actions = jnp.squeeze(distribution.sample(seed=key))
            log_prob = jnp.squeeze(distribution.log_prob(actions))

            return actions, log_prob, policy_state  # type: ignore

        self.forward_fn = forward_fn

    def get_action(
        self,
        observations: networks_lib.Observation,
        params: Any,
        key: networks_lib.PRNGKey,
        mask: chex.Array = None,
        policy_state: Any = None,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Get actions from policy network given observations."""

        actions, log_prob, policy_state = self.forward_fn(
            policy_params=params["policy_network"],
            observations=observations,
            key=key,
            mask=mask,
            policy_state=policy_state,
        )

        if self.policy_init_state:
            return actions, {"log_prob": log_prob}, policy_state  # type: ignore
        else:
            return actions, {"log_prob": log_prob}  # type: ignore

    def get_value(self, observations: networks_lib.Observation) -> jnp.ndarray:
        """Get state value from critic network given observations."""
        value = self.critic_network.apply(self.critic_params, observations)
        return value

    def get_init_state(self) -> jnp.ndarray:
        return self.policy_init_state

    def get_params(
        self,
    ) -> Dict[str, jnp.ndarray]:
        """Return current params.

        Returns:
            policy and critic params.
        """
        return {
            "policy_network": self.policy_params,
            "critic_network": self.critic_params,
        }


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


def make_discrete_networks(
    environment_spec: specs.EnvironmentSpec,
    base_key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    critic_layer_sizes: Sequence[int],
    policy_recurrent_layer_sizes: Sequence[int],
    recurrent_architecture_fn: Any,
    policy_layers_after_recurrent: Sequence[int],
    # default behaviour is to flatten observations
) -> PPONetworks:
    """Create PPO network for environments with discrete action spaces.

    Args:
        environment_spec: environment spec
        key: jax random key for initializing network parameters
        policy_layer_sizes: sizes of hidden layers for the policy network
        critic_layer_sizes: sizes of hidden layers for the critic network
        policy_recurrent_layer_sizes: Optionally add recurrent layers to the policy
        recurrent_architecture_fn: Architecture to use for the recurrent units.
        policy_layers_after_recurrent: sizes of hidden layers for the policy network after the recurrent layers.
        This is only used if an recurrent architecture is used.
    Returns:
        PPONetworks class
    """

    num_actions = environment_spec.actions.num_values

    @hk.without_apply_rng
    @hk.transform
    def policy_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        # Add the observation network and an MLP network.
        policy_network = [
            hk.nets.MLP(policy_layer_sizes, activation=jax.nn.relu),
        ]

        # Add optional recurrent layers
        if len(policy_recurrent_layer_sizes) > 0:
            for size in policy_recurrent_layer_sizes:
                policy_network.append(recurrent_architecture_fn(size))

            # Add optional feedforward layers after the recurrent layers
            hk.nets.MLP(policy_layers_after_recurrent, activation=jax.nn.relu),

        # Add a categorical value head.
        policy_network.append(
            networks_lib.CategoricalHead(
                num_values=num_actions, dtype=environment_spec.actions.dtype
            )
        )

        if len(policy_recurrent_layer_sizes) > 0:
            return hk.DeepRNN(policy_network)(inputs[0], inputs[1])
        else:
            return hk.Sequential(policy_network)(inputs)

    @hk.without_apply_rng
    @hk.transform
    def initial_state_fn() -> List[jnp.ndarray]:
        state = []
        for size in policy_recurrent_layer_sizes:
            state.append(recurrent_architecture_fn(size).initial_state(1))
        return state

    @hk.without_apply_rng
    @hk.transform
    def critic_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        critic_network = hk.Sequential(
            [
                hk.nets.MLP(critic_layer_sizes, activation=jax.nn.relu),
                ValueHead(),
            ]
        )
        return critic_network(inputs)

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    base_key, network_key = jax.random.split(base_key)

    if len(policy_recurrent_layer_sizes) > 0:
        policy_state = initial_state_fn.apply(None)

        policy_params = policy_fn.init(network_key, [dummy_obs, policy_state])  # type: ignore
    else:
        policy_state = None
        policy_params = policy_fn.init(network_key, dummy_obs)  # type: ignore

    base_key, network_key = jax.random.split(base_key)
    critic_params = critic_fn.init(network_key, dummy_obs)  # type: ignore

    # Create PPONetworks to add functionality required by the agent.
    return PPONetworks(
        policy_network=policy_fn,
        policy_params=policy_params,
        policy_init_state=policy_state,
        critic_network=critic_fn,
        critic_params=critic_params,
        log_prob=lambda distribution, action: distribution.log_prob(action),
        entropy=lambda distribution: distribution.entropy(),
        sample=lambda distribution, key: distribution.sample(seed=key),
    )


def make_networks(
    spec: specs.EnvironmentSpec,
    base_key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    critic_layer_sizes: Sequence[int],
    policy_recurrent_layer_sizes: Sequence[int],
    recurrent_architecture_fn: Any,
    policy_layers_after_recurrent: Sequence[int],
) -> PPONetworks:
    """Function for creating PPO networks to be used.

    These networks will be different depending on whether the
    environment has a discrete or continuous action space.

    Args:
        spec: specifications of training environment
        base_key: pseudo-random value used to initialise distributions
        policy_layer_sizes: size of each layer of the policy network
        critic_layer_sizes: size of each layer of the critic network
        policy_recurrent_layer_sizes: Optionally add recurrent layers to the policy
        recurrent_architecture_fn: Architecture to use for the recurrent units
        sizes of hidden layers for the policy network after the recurrent layers.
        This is only used if an recurrent architecture is used.

    Returns:
        make_discrete_networks: function to create a discrete network

    Raises:
        NotImplementedError: Raises an error if continous network is not
                        available
    """
    if isinstance(spec.actions, specs.DiscreteArray):
        return make_discrete_networks(
            environment_spec=spec,
            base_key=base_key,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            policy_layers_after_recurrent=policy_layers_after_recurrent,
            policy_recurrent_layer_sizes=policy_recurrent_layer_sizes,
            recurrent_architecture_fn=recurrent_architecture_fn,
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
        256,
        256,
        256,
    ),
    critic_layer_sizes: Sequence[int] = (512, 512, 256),
    policy_recurrent_layer_sizes: Sequence[int] = (),
    recurrent_architecture_fn: Any = hk.GRU,
    policy_layers_after_recurrent: Sequence[int] = (),
) -> Dict[str, Any]:
    """Create default PPO networks

    Args:
        environment_spec: mava multi-agent environment spec
        agent_net_keys: dictionary specifiying which networks are
                        used by which agent
        base_key: jax random key to be used for network initialization
        net_spec_keys: keys for each agent network
        policy_layer_sizes: policy network layers
        critic_layer_sizes: critic network layers
        policy_recurrent_layer_sizes: Optionally add recurrent layers to the policy
        recurrent_architecture_fn: Architecture to use for the recurrent units
        sizes of hidden layers for the policy network after the recurrent layers.
        This is only used if an recurrent architecture is used.
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
            critic_layer_sizes=critic_layer_sizes,
            policy_recurrent_layer_sizes=policy_recurrent_layer_sizes,
            recurrent_architecture_fn=recurrent_architecture_fn,
            policy_layers_after_recurrent=policy_layers_after_recurrent,
        )

    # No longer returning a dictionary since this is handled in PPONetworks above
    return networks
