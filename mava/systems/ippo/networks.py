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
from mava.types import NestedArray
from mava.utils.jax_training_utils import action_mask_categorical_policies
from mava.utils.networks_utils import MLP_NORM

Array = dm_specs.Array
BoundedArray = dm_specs.BoundedArray
DiscreteArray = dm_specs.DiscreteArray
EntropyFn = Callable[[Any], jnp.ndarray]


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
            policy_init_state: initial policy hidden state.
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
            rng_key: networks_lib.PRNGKey,
            mask: chex.Array = None,
            policy_state: Tuple[jnp.ndarray] = None,
            evaluate: bool = False,
        ) -> Tuple[NestedArray, NestedArray, NestedArray]:
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
                logits = self.policy_network.apply(policy_params, observations)
            else:
                logits, policy_state = self.policy_network.apply(
                    policy_params, [observations, policy_state]
                )

            if mask is not None:
                distribution = action_mask_categorical_policies(logits, mask)

            # actions = jax.lax.cond(
            #     evaluate,
            #     lambda dist: dist.mode(),
            #     lambda dist: dist.sample(seed=rng_key),
            #     distribution,
            # )

            actions = distribution.sample(seed=rng_key)
            log_prob = jnp.squeeze(distribution.log_prob(actions))

            return jnp.squeeze(actions), log_prob, policy_state

        self.forward_fn = forward_fn

    def get_action(
        self,
        observations: networks_lib.Observation,
        params: Any,
        base_key: networks_lib.PRNGKey,
        mask: chex.Array = None,
        policy_state: Any = None,
        evaluate: bool = False,
    ) -> Tuple[jnp.ndarray, Dict]:
        """Get actions from policy network given observations."""

        actions, log_prob, policy_state = self.forward_fn(
            policy_params=params["policy_network"],
            observations=observations,
            rng_key=base_key,
            mask=mask,
            policy_state=policy_state,
            evaluate=evaluate,
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
        """Get initial policy hidden state."""
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
        w_init: Optional[hk.initializers.Initializer] = None,
    ):
        """Initialize the value head.

        Args:
            name: An optional string name for the class.
            w_init: Initializer for network weights.
        """
        super().__init__(name=name)
        self._value_layer = hk.Linear(1, w_init=w_init)

    def __call__(self, inputs: jnp.ndarray) -> Any:
        """Return output given network inputs."""
        value = jnp.squeeze(self._value_layer(inputs), axis=-1)
        return value


def make_discrete_networks(
    environment_spec: specs.EnvironmentSpec,
    num_agents: int,
    base_key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    critic_layer_sizes: Sequence[int],
    policy_recurrent_layer_sizes: Sequence[int],
    recurrent_architecture_fn: Any,
    policy_layers_after_recurrent: Sequence[int],
    observation_network: Optional[Callable] = None,
    orthogonal_initialisation: bool = False,
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    policy_network_head_weight_gain: float = 0.01,
    layer_norm: bool = False,
) -> PPONetworks:
    """Create PPO network for environments with discrete action spaces.

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
        layer_norm: apply layer normalisation to the hidden MLP layers.


    Returns:
        PPONetworks class
    """

    num_actions = environment_spec.actions.num_values

    # Define weight and bias initialisation functions to be
    # used in MLPs and policy and critic head networks.
    w_init_fn = (
        lambda x, scale: hk.initializers.Orthogonal(scale=scale)
        if (x is True)
        else None
    )

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
        policy_layers = []
        if observation_network:
            policy_layers.extend([observation_network])
        policy_layers.extend(
            [
                MLP_NORM(
                    policy_layer_sizes,
                    activation=activation_function,
                    w_init=w_init_fn(orthogonal_initialisation, jnp.sqrt(2)),
                    activate_final=True,
                    layer_norm=layer_norm,
                )
            ]
        )

        # Add optional recurrent layers
        if len(policy_recurrent_layer_sizes) > 0:
            for size in policy_recurrent_layer_sizes:
                policy_layers.append(recurrent_architecture_fn(size))

            # Add optional feedforward layers after the recurrent layers
            if len(policy_layers_after_recurrent) > 0:
                policy_layers.append(
                    MLP_NORM(
                        policy_layers_after_recurrent,
                        activation=activation_function,
                        w_init=w_init_fn(orthogonal_initialisation, jnp.sqrt(2)),
                        activate_final=True,
                        layer_norm=layer_norm,
                    ),
                )

        # Add a categorical value head.
        policy_layers.append(
            networks_lib.CategoricalHead(
                num_values=num_actions,
                dtype=environment_spec.actions.dtype,
                w_init=w_init_fn(
                    orthogonal_initialisation, policy_network_head_weight_gain
                ),
            )
        )

        if len(policy_recurrent_layer_sizes) > 0:
            return hk.DeepRNN(policy_layers)(inputs[0], inputs[1])
        else:
            return hk.Sequential(policy_layers)(inputs)

    @hk.without_apply_rng
    @hk.transform
    def initial_state_fn() -> List[jnp.ndarray]:
        """Returns an intial state for the recurrent layers.

        Args:
            None.

        Returns:
            Initial state for the recurrent layers.
        """
        state = []
        for size in policy_recurrent_layer_sizes:
            state.append(recurrent_architecture_fn(size).initial_state(1))
        return state

    @hk.without_apply_rng
    @hk.transform
    def critic_fn(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        """Create a critic network function and transform it using Haiku.

        Args:
            inputs: The inputs required for hk.Sequential.

        Returns:
            FeedForwardNetwork class
        """
        critic_layers = []
        if observation_network:
            critic_layers.extend([observation_network])
        critic_layers.extend(
            [
                MLP_NORM(
                    critic_layer_sizes,
                    activation=activation_function,
                    w_init=w_init_fn(orthogonal_initialisation, jnp.sqrt(2)),
                    activate_final=True,
                    layer_norm=layer_norm,
                ),
                ValueHead(w_init=w_init_fn(orthogonal_initialisation, 1.0)),
            ]
        )
        critic_network = hk.Sequential(critic_layers)
        return critic_network(inputs)

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    # replicate obs num agents times
    dummy_concatted_obs = jnp.concatenate(
        [dummy_obs for _ in range(num_agents)], axis=1
    )

    base_key, network_key = jax.random.split(base_key)

    if len(policy_recurrent_layer_sizes) > 0:
        policy_state = initial_state_fn.apply(None)

        policy_params = policy_fn.init(network_key, [dummy_obs, policy_state])  # type: ignore # noqa: E501
    else:
        policy_state = None
        policy_params = policy_fn.init(network_key, dummy_obs)  # type: ignore

    base_key, network_key = jax.random.split(base_key)
    critic_params = critic_fn.init(network_key, dummy_concatted_obs)  # type: ignore

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
    num_agents: int,
    base_key: networks_lib.PRNGKey,
    policy_layer_sizes: Sequence[int],
    critic_layer_sizes: Sequence[int],
    policy_recurrent_layer_sizes: Sequence[int],
    recurrent_architecture_fn: Any,
    policy_layers_after_recurrent: Sequence[int],
    observation_network: Optional[Callable] = None,
    orthogonal_initialisation: bool = False,
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    policy_network_head_weight_gain: float = 0.01,
    layer_norm: bool = False,
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
        observation_network: optional network for processing observations.
            Defaults to utils.batch_concat.
        layer_norm: apply layer normalisation to the hidden MLP layers.

    Returns:
        make_discrete_networks: function to create a discrete network

    Raises:
        NotImplementedError: Raises an error if continuous network is not
                        available
    """
    if isinstance(spec.actions, specs.DiscreteArray):
        return make_discrete_networks(
            environment_spec=spec,
            num_agents=num_agents,
            base_key=base_key,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            policy_recurrent_layer_sizes=policy_recurrent_layer_sizes,
            policy_layers_after_recurrent=policy_layers_after_recurrent,
            recurrent_architecture_fn=recurrent_architecture_fn,
            orthogonal_initialisation=orthogonal_initialisation,
            activation_function=activation_function,
            layer_norm=layer_norm,
            policy_network_head_weight_gain=policy_network_head_weight_gain,
            observation_network=observation_network,
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
    orthogonal_initialisation: bool = False,
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    policy_network_head_weight_gain: float = 0.01,
    observation_network: Optional[Callable] = None,
    layer_norm: bool = False,
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
        layer_norm: apply layer normalisation to the hidden MLP layers.

        observation_network: optional network for processing observations.
            Defaults to utils.batch_concat.
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
    num_agents = len(environment_spec.get_agent_environment_specs())
    for net_key in specs.keys():
        networks[net_key] = make_networks(
            specs[net_key],
            num_agents=num_agents,
            base_key=base_key,
            policy_layer_sizes=policy_layer_sizes,
            critic_layer_sizes=critic_layer_sizes,
            policy_recurrent_layer_sizes=policy_recurrent_layer_sizes,
            recurrent_architecture_fn=recurrent_architecture_fn,
            policy_layers_after_recurrent=policy_layers_after_recurrent,
            orthogonal_initialisation=orthogonal_initialisation,
            activation_function=activation_function,
            layer_norm=layer_norm,
            policy_network_head_weight_gain=policy_network_head_weight_gain,
            observation_network=observation_network,
        )

    # No longer returning a dictionary since this is handled in PPONetworks above
    return networks
