"""Generic executor implementations, using TensorFlow and Sonnet."""
from typing import Any, Dict, Optional, Tuple, Union

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme import types

# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders, core
from mava.components.tf.modules.communication import BaseCommunicationModule

tfd = tfp.distributions


class FeedForwardExecutor(core.Executor):
    """A feed-forward executor.
    An executor based on a feed-forward policy for each agent in the system
    which takes non-batched observations and outputs non-batched actions.
    It also allows adding experiences to replay and updating the weights
    from the policy on the learner.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        shared_weights: bool = True,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):
        """Initializes the executor.
        Args:
          policy_network: the policy to run for each agent in the system.
          shared_weights: specify if weights are shared between agent networks.
          adder: the adder object to which allows to add experiences to a
            dataset/replay buffer.
          variable_client: object which allows to copy weights from the trainer copy
            of the policies to the executor copy (in case they are separate).
        """

        # Store these for later use.
        self._shared_weights = shared_weights
        self._policy_networks = policy_networks
        self._adder = adder
        self._variable_client = variable_client

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
    ) -> Union[types.NestedArray, Tuple[types.NestedArray, types.NestedArray]]:
        # Pass the observation through the policy network.
        action = self._policy(agent, observation.observation)

        # TODO Mask actions here using observation.legal_actions
        # What happens in discrete vs cont case

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(action)

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:

        if self._adder:
            self._adder.add_first(timestep, extras)

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        if self._adder:
            self._adder.add(actions, next_timestep, next_extras)

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Union[
        Dict[str, types.NestedArray],
        Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]],
    ]:
        actions = {}
        for agent, observation in observations.items():
            # Pass the observation through the policy network.
            action = self._policy(agent, observation.observation)
            # TODO Mask actions here using observation.legal_actions
            # What happens in discrete vs cont case
            actions[agent] = tf2_utils.to_numpy_squeeze(action)

        # Return a numpy array with squeezed out batch dimension.
        return actions

    def update(self, wait: bool = False) -> None:
        if self._variable_client:
            self._variable_client.update(wait)


class RecurrentExecutor(core.Executor):
    """A recurrent Executor.
    An executor based on a recurrent policy for each agent in the system which
    takes non-batched observations and outputs non-batched actions, and keeps
    track of the recurrent state inside. It also allows adding experiences to
    replay and updating the weights from the policy on the learner.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.RNNCore],
        shared_weights: bool = True,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
    ):
        """Initializes the executor.
        Args:
          policy_networks: the (recurrent) policy to run for each agent in the system.
          shared_weights: specify if weights are shared between agent networks.
          adder: the adder object to which allows to add experiences to a
            dataset/replay buffer.
          variable_client: object which allows to copy weights from the trainer copy
            of the policies to the executor copy (in case they are separate).
          store_recurrent_state: Whether to pass the recurrent state to the adder.
        """
        # Store these for later use.
        self._policy_networks = policy_networks
        self._shared_weights = shared_weights
        self._adder = adder
        self._variable_client = variable_client
        self._store_recurrent_state = store_recurrent_state
        self._states: Dict[str, Any] = {}

    @tf.function
    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        state: types.NestedTensor,
    ) -> Union[
        Tuple[types.NestedTensor, types.NestedTensor],
        Tuple[types.NestedTensor, types.NestedTensor, types.NestedTensor],
    ]:

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # index network either on agent type or on agent id
        agent_key = agent.split("_")[0] if self._shared_weights else agent

        # Compute the policy, conditioned on the observation.
        policy, new_state = self._policy_networks[agent_key](batched_observation, state)

        # Sample from the policy if it is stochastic.
        action = policy.sample() if isinstance(policy, tfd.Distribution) else policy

        return action, new_state

    def _update_state(self, agent: str, new_state: types.NestedArray) -> None:
        # Bookkeeping of recurrent states for the observe method.
        self._states[agent] = new_state

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:

        # TODO Mask actions here using observation.legal_actions
        # What happens in discrete vs cont case

        # Initialize the RNN state if necessary.
        if self._states[agent] is None:
            # index network either on agent type or on agent id
            agent_key = agent.split("_")[0] if self._shared_weights else agent
            self._states[agent] = self._policy_networks[agent_key].initial_state(1)

        # Step the recurrent policy forward given the current observation and state.
        policy_output, new_state = self._policy(
            agent, observation.observation, self._states[agent]
        )

        # Bookkeeping of recurrent states for the observe method.
        self._update_state(agent, new_state)

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(policy_output)

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Optional[Dict[str, types.NestedArray]] = {},
    ) -> None:
        # Re-initialize the RNN state.
        for agent, _ in timestep.observation.items():
            # index network either on agent type or on agent id
            agent_key = agent.split("_")[0] if self._shared_weights else agent
            self._states[agent] = self._policy_networks[agent_key].initial_state(1)

        if self._adder is not None:
            numpy_states = {
                agent: tf2_utils.to_numpy_squeeze(_state)
                for agent, _state in self._states.items()
            }

            if extras:
                extras.update({"core_states": numpy_states})
                self._adder.add_first(timestep, extras)
            else:
                self._adder.add_first(timestep, numpy_states)

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Optional[Dict[str, types.NestedArray]] = {},
    ) -> None:
        if not self._adder:
            return

        if not self._store_recurrent_state:
            if next_extras:
                self._adder.add(actions, next_timestep, next_extras)
            else:
                self._adder.add(actions, next_timestep)
            return

        numpy_states = {
            agent: tf2_utils.to_numpy_squeeze(_state)
            for agent, _state in self._states.items()
        }
        if next_extras:
            next_extras.update({"core_states": numpy_states})
            self._adder.add(actions, next_timestep, next_extras)
        else:
            self._adder.add(actions, next_timestep, numpy_states)

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Union[
        Dict[str, types.NestedArray],
        Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]],
    ]:
        actions = {}
        for agent, observation in observations.items():
            # Step the recurrent policy forward given the current observation and state.
            actions[agent] = self.select_action(agent=agent, observation=observation)
        return actions

    def update(self, wait: bool = False) -> None:
        if self._variable_client:
            self._variable_client.update(wait)


class RecurrentCommExecutor(RecurrentExecutor):
    """Recurrent executor where agents can communicate."""

    def __init__(
        self,
        policy_networks: Dict[str, snt.RNNCore],
        communication_module: BaseCommunicationModule,
        shared_weights: bool = True,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
    ):
        """Initializes the executor.
        TODO: Update docstring
        Args:
          policy_networks: the (recurrent) policy to run for each agent in the system.
          shared_weights: specify if weights are shared between agent networks.
          adder: the adder object to which allows to add experiences to a
            dataset/replay buffer.
          variable_client: object which allows to copy weights from the trainer copy
            of the policies to the executor copy (in case they are separate).
          store_recurrent_state: Whether to pass the recurrent state to the adder.
        """
        # Store these for later use.

        self._adder = adder
        self._variable_client = variable_client
        self._policy_networks = policy_networks
        self._states: Dict[str, Any] = {}
        self._messages: Dict[str, Any] = {}
        self._store_recurrent_state = store_recurrent_state
        self._communication_module = communication_module
        self._shared_weights = shared_weights

    def _sample_action(
        self, action_policy: types.NestedTensor, agent: str
    ) -> types.NestedTensor:
        # Sample from the policy if it is stochastic.
        return (
            action_policy.sample()
            if isinstance(action_policy, tfd.Distribution)
            else action_policy
        )

    def _process_message(
        self, observation: types.NestedTensor, message_policy: types.NestedTensor
    ) -> types.NestedTensor:
        # Only one agent can message at each timestep
        return message_policy

    @tf.function
    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        state: types.NestedTensor,
        message: types.NestedTensor,
    ) -> Tuple[types.NestedTensor, types.NestedTensor, types.NestedTensor]:

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # index network either on agent type or on agent id
        agent_key = agent.split("_")[0] if self._shared_weights else agent

        # Compute the policy, conditioned on the observation.
        (action_policy, message_policy), new_state = self._policy_networks[agent_key](
            batched_observation, state, message
        )

        action = self._sample_action(action_policy, agent)

        message = self._process_message(observation, message_policy)

        return action, message, new_state

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Optional[Dict[str, types.NestedArray]] = {},
    ) -> None:

        # Re-initialize the RNN state.
        for agent, _ in timestep.observation.items():
            # index network either on agent type or on agent id
            agent_key = agent.split("_")[0] if self._shared_weights else agent
            self._states[agent] = self._policy_networks[agent_key].initial_state(1)
            self._messages[agent] = self._policy_networks[agent_key].initial_message(1)

        if self._adder is not None:
            numpy_states = {
                agent: tf2_utils.to_numpy_squeeze(_state)
                for agent, _state in self._states.items()
            }
            numpy_messages = {
                agent: tf2_utils.to_numpy_squeeze(_message)
                for agent, _message in self._messages.items()
            }
            if extras is not None:
                extras.update(
                    {
                        "core_states": numpy_states,
                        "core_messages": numpy_messages,
                    }
                )
                self._adder.add_first(timestep, extras)
            else:
                raise NotImplementedError("Why is extras None?")

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Optional[Dict[str, types.NestedArray]] = {},
    ) -> None:
        if not self._adder:
            return

        if not self._store_recurrent_state:
            if next_extras:
                self._adder.add(actions, next_timestep, next_extras)
            else:
                self._adder.add(actions, next_timestep)
            return

        numpy_states = {
            agent: tf2_utils.to_numpy_squeeze(_state)
            for agent, _state in self._states.items()
        }
        numpy_messages = {
            agent: tf2_utils.to_numpy_squeeze(_message)
            for agent, _message in self._messages.items()
        }

        if next_extras:
            next_extras.update(
                {
                    "core_states": numpy_states,
                    "core_messages": numpy_messages,
                }
            )
            self._adder.add(actions, next_timestep, next_extras)
        else:
            next_extras = {
                "core_states": numpy_states,
                "core_messages": numpy_messages,
            }
            self._adder.add(actions, next_timestep, next_extras)

    def select_action(
        self,
        agent: str,
        observation: types.NestedArray,
    ) -> Tuple[types.NestedArray, types.NestedArray]:
        """Get actions and messages of specific agent"""

        # Initialize the RNN state if necessary.
        if self._states[agent] is None:
            self._states[agent] = self._networks[agent].initial_state(1)

        message_inputs = self._communication_module.process_messages(self._messages)
        # Step the recurrent policy forward given the current observation and state.
        policy_output, message, new_state = self._policy(
            agent,
            observation.observation,
            self._states[agent],
            message_inputs[agent],
        )

        # Bookkeeping of recurrent states for the observe method.
        self._states[agent] = new_state
        self._messages[agent] = message

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(policy_output)

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Dict[str, types.NestedArray]:
        actions = {}
        for agent, observation in observations.items():
            actions[agent] = self._select_action(agent, observations)
        # Return a numpy array with squeezed out batch dimension.
        return actions
