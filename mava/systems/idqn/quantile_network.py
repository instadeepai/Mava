import copy
import dataclasses
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import haiku as hk  # type: ignore
import jax
import jax.numpy as jnp
from acme import specs
from acme.jax import networks as networks_lib
from acme.jax import utils
from distrax import Categorical

from mava import specs as mava_specs


@dataclasses.dataclass
class QuantileRegressionNetwork:
    def __init__(
        self,
        policy_params: networks_lib.Params,
        network: networks_lib.FeedForwardNetwork,
    ) -> None:
        """A container for IDQN networks.

        Holds target and main network

        Args:
            policy_params: parameters of the policy network
            network: structure of the policy network

        Return:
            IDQNNetwork
        """
        self.policy_params: networks_lib.Params = policy_params
        self.target_policy_params: networks_lib.Params = copy.deepcopy(policy_params)
        self.policy_network: networks_lib.FeedForwardNetwork = network

        def forward_fn(
            policy_params: Dict[str, jnp.ndarray],
            observations: networks_lib.Observation,
        ) -> jnp.ndarray:
            """Get Q values from the network given observations

            Args:
                policy_params: parameters of the policy network
                observations: agent observations

            Returns: Q-values of all actions in the current state
            """
            return self.policy_network.apply(policy_params, observations)

        self.forward = forward_fn

    def get_action(
        self,
        params: networks_lib.Params,
        observations: networks_lib.Observation,
        epsilon: float,
        base_key: jax.random.KeyArray,
        mask: jnp.ndarray,
    ) -> jnp.ndarray:
        """Get actions from policy network given observations.

        Args:
            policy_params: parameters of the policy network
            observations: agent observations
            epsilon: probability that the agent takes a random action
            base_key: jax random key
            mask: action mask of the legal actions

        Returns:
            an action to take in the current state
        """
        q_values, _ = self.forward(params, observations)
        masked_q_values = jnp.where(mask == 1.0, q_values, jnp.finfo(jnp.float32).min)

        greedy_actions = masked_q_values == jnp.max(masked_q_values)
        greedy_actions_probs = greedy_actions / jnp.sum(greedy_actions)

        random_action_probs = mask / jnp.sum(mask)

        weighted_gready_probs = (1 - epsilon) * greedy_actions_probs
        weighted_rand_probs = epsilon * random_action_probs
        combined_probs = weighted_gready_probs + weighted_rand_probs

        action_dist = Categorical(probs=combined_probs)
        return action_dist.sample(seed=base_key)

    def get_params(
        self,
    ) -> Dict[str, networks_lib.Params]:
        """Return current params of the target and policy network.

        Returns:
            policy and target policy params.
        """
        return {
            "policy_network": self.policy_params,
            "target_policy_network": self.target_policy_params,
        }


def _make_quantile_network(
    environment_spec: specs.EnvironmentSpec,
    base_key: jax.random.KeyArray,
    policy_layer_sizes: Sequence[int] = (512,),
    num_atoms: int = 51,
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    observation_network: Optional[Callable] = None,
) -> QuantileRegressionNetwork:
    # TODO: why?
    # assert (
    #     len(policy_layer_sizes) == 1
    # ), "QR DQN doesn't seem to work with more than 1 layer in the Q network"
    num_actions = environment_spec.actions.num_values

    @hk.without_apply_rng
    @hk.transform
    def q_function(inputs: jnp.ndarray) -> networks_lib.FeedForwardNetwork:
        model = hk.nets.MLP(
            [
                *policy_layer_sizes,
                num_actions * num_atoms,
            ],
            activation=activation_function,
        )

        # Add obs net
        # if observation_network is not None:
        #     model = [observation_network, model]
        # else:
        #     model = [model]

        # model = hk.Sequential(model)

        q_dist = model(inputs).reshape(
            -1, environment_spec.actions.num_values, num_atoms
        )
        q_values = jnp.mean(q_dist, axis=-1)

        return q_values, q_dist

    dummy_obs = utils.zeros_like(environment_spec.observations.observation)
    dummy_obs = utils.add_batch_dim(dummy_obs)  # Dummy 'sequence' dim.

    base_key, network_key = jax.random.split(base_key)

    policy_params = q_function.init(network_key, dummy_obs)

    base_key, network_key = jax.random.split(base_key)

    return QuantileRegressionNetwork(
        network=q_function,
        policy_params=policy_params,
    )


def make_quantile_regression_networks(
    environment_spec: mava_specs.MAEnvironmentSpec,
    agent_net_keys: Dict[str, str],
    base_key: jax.random.KeyArray,
    net_spec_keys: Dict[str, str] = {},
    policy_layer_sizes: Sequence[int] = (512,),
    num_atoms: int = 51,
    activation_function: Callable[[jnp.ndarray], jnp.ndarray] = jax.nn.relu,
    observation_network: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Create quantile regression IDQN networks (one per agent)

    Args:
        environment_spec: mava multi-agent environment spec
        agent_net_keys: dictionary specifying which networks are
            used by which agent
        base_key: jax random key to be used for network initialization
        net_spec_keys: keys for each agent network
        num_atoms: the number of quantiles to use
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
        networks[net_key] = _make_quantile_network(
            specs[net_key],
            base_key=base_key,
            policy_layer_sizes=policy_layer_sizes,
            activation_function=activation_function,
            observation_network=observation_network,
            num_atoms=num_atoms,
        )

    return networks
