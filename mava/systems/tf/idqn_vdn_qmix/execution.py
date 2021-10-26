from typing import Optional

import numpy as np
import tensorflow as tf
from acme import types
from acme.tf import utils as tf2_utils

from mava import adders
from mava.components.tf.networks.epsilon_greedy import epsilon_greedy_action_selector

class FeedForwardExecutor:

    def __init__(
        self,
        qnetwork,
        epsilon_scheduler,
        variable_client,
        adder = None
    ):
        # Store these for later use.
        self._qnetwork = qnetwork
        self._epsilon_scheduler = epsilon_scheduler
        self._variable_client = variable_client
        self._adder = adder

    @tf.function
    def _policy(
        self,
        observation: tf.Tensor,
        legals: tf.Tensor,
        epsilon: tf.Tensor
    ):
        # Compute action values 
        action_values = self._qnetwork(observation)

        # Epsilon Greedy Action Selection
        action = epsilon_greedy_action_selector(action_values, epsilon, legal_actions_mask=legals)

        return action

    def select_actions(
        self, observations, evaluator: bool = False
    ):
        # Get Epsilon
        if evaluator:
            # No exploration.
            epsilon = 0.0
        else:
            # Get and decrement epsilon.
            self._epsilon_scheduler.decrement_epsilon()
            epsilon = self._epsilon_scheduler.get_epsilon()
        epsilon = tf.cast(tf.convert_to_tensor(epsilon), dtype='float32')
        
        actions = {}
        for agent, observation in observations.items():

            batched_obs = tf2_utils.add_batch_dim(observation.observation)
            batched_legals = tf2_utils.add_batch_dim(observation.legal_actions)

            action = self._policy(
                batched_obs,
                batched_legals,
                epsilon,
            )

            actions[agent] = action

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(actions)

    def observe_first(
        self,
        timestep,
        extras = {},
    ):
        if self._adder is not None:
            self._adder.add_first(timestep, extras)

    def observe(
        self,
        actions,
        next_timestep,
        next_extras = {},
    ) -> None:

        if not self._adder:
            return

        self._adder.add(actions, next_timestep, next_extras)

    def update(self, wait: bool = False) -> None:
        self._variable_client.update(wait)

class RecurrentExecutor(FeedForwardExecutor):

    def __init__(
        self,
        qnetwork,
        epsilon_scheduler,
        variable_client,
        adder: Optional[adders.ParallelAdder] = None,
    ):
        super().__init__(
            qnetwork,
            epsilon_scheduler,
            variable_client,
            adder
        )

        # Recurrent states
        self._states = {}

    @tf.function
    def _policy(
        self,
        observation: tf.Tensor,
        legals: tf.Tensor,
        state: tf.Tensor,
        epsilon: tf.Tensor
    ):
        # Compute action values 
        action_values, state = self._qnetwork(observation, state)

        # Epsilon Greedy Action Selection
        # TODO add legal action mask
        action = epsilon_greedy_action_selector(action_values, epsilon, legal_actions_mask=legals)

        return action, state

    def select_actions(
        self, observations, evaluator: bool = False
    ):
        # Get Epsilon
        if evaluator:
            # No exploration.
            epsilon = 0.0
        else:
            # Get and decrement epsilon.
            self._epsilon_scheduler.decrement_epsilon()
            epsilon = self._epsilon_scheduler.get_epsilon()
        epsilon = tf.convert_to_tensor(epsilon)

        actions = {}
        for agent, observation in observations.items():

            batched_obs = tf2_utils.add_batch_dim(observation.observation)
            batched_legals = tf2_utils.add_batch_dim(observation.legal_actions)

            action, state = self._policy(
                batched_obs,
                batched_legals,
                self._states[agent],
                epsilon,
            )

            actions[agent] = action
            self._states[agent] = state

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(actions)

    def observe_first(
        self,
        timestep,
        extras = {},
    ):

        # Re-initialize the RNN state.
        for agent, _ in timestep.observation.items():
            self._states[agent] = self._qnetwork.initial_state(1)

        if self._adder is not None:
            numpy_states = {
                agent: tf2_utils.to_numpy_squeeze(_state)
                for agent, _state in self._states.items()
            }

            extras.update({"core_states": numpy_states})
            self._adder.add_first(timestep, extras)

    def observe(
        self,
        actions,
        next_timestep,
        next_extras = {},
    ) -> None:

        if not self._adder:
            return

        numpy_states = {
            agent: tf2_utils.to_numpy_squeeze(_state)
            for agent, _state in self._states.items()
        }

        next_extras.update({"core_states": numpy_states})
        self._adder.add(actions, next_timestep, next_extras)
   