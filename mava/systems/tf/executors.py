"""Generic executor implementations, using TensorFlow and Sonnet."""

from typing import Dict, Optional

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme import types

# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders, core

tfd = tfp.distributions


class FeedForwardExecutor(core.Executor):
    """A feed-forward actor.
    An actor based on a feed-forward policy which takes non-batched observations
    and outputs non-batched actions. It also allows adding experiences to replay
    and updating the weights from the policy on the learner.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        adder: Optional[adders.ParallelAdder] = None,
        variable_clients: Optional[Dict[str, tf2_variable_utils.VariableClient]] = None,
    ):
        """Initializes the actor.
        Args:
          policy_network: the policy to run.
          adder: the adder object to which allows to add experiences to a
            dataset/replay buffer.
          variable_client: object which allows to copy weights from the learner copy
            of the policy to the actor copy (in case they are separate).
        """

        # Store these for later use.
        self._adder = adder
        self._variable_clients = variable_clients
        self._policy_networks = policy_networks
        self._shared_weights = shared_weights

    @tf.function
    def _policy(
        self, agent: str, observation: types.NestedTensor
    ) -> types.NestedTensor:
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # index network either on agent type or on agent id
        agent_key = agent.split("_")[0] if self._shared_weights else agent

        # Compute the policy, conditioned on the observation.
        policy = self._policy_networks[agent_key](batched_observation)

        # Sample from the policy if it is stochastic.
        action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

        return action

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:
        # Pass the observation through the policy network.
        action = self._policy(agent, observation)

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(action)

    def observe_first(self, timestep: dm_env.TimeStep) -> None:
        if self._adder:
            self._adder.add_first(timestep)

    def observe(
        self, actions: Dict[str, types.NestedArray], next_timestep: dm_env.TimeStep
    ) -> None:
        if self._adder:
            self._adder.add(actions, next_timestep)

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Dict[str, types.NestedArray]:
        actions = {}
        for agent, observation in observations.items():
            # Pass the observation through the policy network.
            action = self._policy(agent, observation)
            actions[agent] = tf2_utils.to_numpy_squeeze(action)

        # Return a numpy array with squeezed out batch dimension.
        return actions

    def update(self, wait: bool = False) -> None:
        if self._variable_clients:
            for client in self._variable_clients.values():
                client.update(wait)


# Internal class 1.
