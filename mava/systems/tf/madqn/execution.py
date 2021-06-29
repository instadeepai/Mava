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


"""MADQN system executor implementation."""

from typing import Any, Dict, Optional

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
from acme import types

# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.systems.tf.executors import (
    FeedForwardExecutor,
    RecurrentCommExecutor,
    RecurrentExecutor,
)
from mava.systems.tf.madqn.training import MADQNTrainer
from mava.types import OLT


class MADQNFeedForwardExecutor(FeedForwardExecutor):
    """A feed-forward executor.
    An executor based on a feed-forward policy for each agent in the system
    which takes non-batched observations and outputs non-batched actions.
    It also allows adding experiences to replay and updating the weights
    from the policy on the learner.
    """

    def __init__(
        self,
        q_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        trainer: MADQNTrainer,
        shared_weights: bool = True,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        communication_module: Optional[BaseCommunicationModule] = None,
        fingerprint: bool = False,
        evaluator: bool = False,
    ):
        """[summary]

        Args:
            q_networks (Dict[str, snt.Module]): [description]
            action_selectors (Dict[str, snt.Module]): [description]
            trainer (MADQNTrainer): [description]
            shared_weights (bool, optional): [description]. Defaults to True.
            adder (Optional[adders.ParallelAdder], optional): [description]. Defaults
                to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                [description]. Defaults to None.
            communication_module (Optional[BaseCommunicationModule], optional):
                [description]. Defaults to None.
            fingerprint (bool, optional): [description]. Defaults to False.
            evaluator (bool, optional): [description]. Defaults to False.
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._q_networks = q_networks
        self._action_selectors = action_selectors
        self._trainer = trainer
        self._shared_weights = shared_weights
        self._fingerprint = fingerprint
        self._evaluator = evaluator

    @tf.function
    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        legal_actions: types.NestedTensor,
        epsilon: tf.Tensor,
        fingerprint: Optional[tf.Tensor] = None,
    ) -> types.NestedTensor:
        """[summary]

        Args:
            agent (str): [description]
            observation (types.NestedTensor): [description]
            legal_actions (types.NestedTensor): [description]
            epsilon (tf.Tensor): [description]
            fingerprint (Optional[tf.Tensor], optional): [description].
                Defaults to None.

        Returns:
            types.NestedTensor: [description]
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)
        batched_legals = tf2_utils.add_batch_dim(legal_actions)

        # index network either on agent type or on agent id
        agent_key = agent.split("_")[0] if self._shared_weights else agent

        # Compute the policy, conditioned on the observation and
        # possibly the fingerprint.
        if fingerprint is not None:
            q_values = self._q_networks[agent_key](batched_observation, fingerprint)
        else:
            q_values = self._q_networks[agent_key](batched_observation)

        # select legal action
        action = self._action_selectors[agent_key](
            q_values, batched_legals, epsilon=epsilon
        )

        return action

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:
        """[summary]

        Args:
            agent (str): [description]
            observation (types.NestedArray): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            types.NestedArray: [description]
        """

        if not self._evaluator:
            epsilon = self._trainer.get_epsilon()
        else:
            epsilon = 1e-10

        epsilon = tf.convert_to_tensor(epsilon)

        if self._fingerprint:
            trainer_step = self._trainer.get_trainer_steps()
            fingerprint = tf.concat([epsilon, trainer_step], axis=0)
            fingerprint = tf.expand_dims(fingerprint, axis=0)
            fingerprint = tf.cast(fingerprint, "float32")
        else:
            fingerprint = None

        action = self._policy(
            agent,
            observation.observation,
            observation.legal_actions,
            epsilon,
            fingerprint,
        )

        action = tf2_utils.to_numpy_squeeze(action)

        return action

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """[summary]

        Args:
            timestep (dm_env.TimeStep): [description]
            extras (Dict[str, types.NestedArray], optional): [description].
                Defaults to {}.
        """

        if self._fingerprint and self._trainer is not None:
            epsilon = self._trainer.get_epsilon()
            trainer_step = self._trainer.get_trainer_steps()
            fingerprint = np.array([epsilon, trainer_step])
            extras.update({"fingerprint": fingerprint})

        if self._adder:
            self._adder.add_first(timestep, extras)

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """[summary]

        Args:
            actions (Dict[str, types.NestedArray]): [description]
            next_timestep (dm_env.TimeStep): [description]
            next_extras (Dict[str, types.NestedArray], optional): [description].
                Defaults to {}.
        """

        if self._fingerprint and self._trainer is not None:
            trainer_step = self._trainer.get_trainer_steps()
            epsilon = self._trainer.get_epsilon()
            fingerprint = np.array([epsilon, trainer_step])
            next_extras.update({"fingerprint": fingerprint})

        if self._adder:
            self._adder.add(actions, next_timestep, next_extras)

    def select_actions(
        self, observations: Dict[str, OLT]
    ) -> Dict[str, types.NestedArray]:
        """[summary]

        Args:
            observations (Dict[str, OLT]): [description]

        Returns:
            Dict[str, types.NestedArray]: [description]
        """

        actions = {}
        for agent, observation in observations.items():
            # Pass the observation through the policy network.
            if not self._evaluator:
                epsilon = self._trainer.get_epsilon()
            else:
                # Note (dries): For some reason 0 epsilon breaks on StarCraft.
                epsilon = 1e-10

            epsilon = tf.convert_to_tensor(epsilon)

            if self._fingerprint:
                trainer_step = self._trainer.get_trainer_steps()
                fingerprint = tf.concat([epsilon, trainer_step], axis=0)
                fingerprint = tf.expand_dims(fingerprint, axis=0)
                fingerprint = tf.cast(fingerprint, "float32")
            else:
                fingerprint = None

            action = self._policy(
                agent,
                observation.observation,
                observation.legal_actions,
                epsilon,
                fingerprint,
            )

            actions[agent] = tf2_utils.to_numpy_squeeze(action)

        # Return a numpy array with squeezed out batch dimension.
        return actions

    def update(self, wait: bool = False) -> None:
        """[summary]

        Args:
            wait (bool, optional): [description]. Defaults to False.
        """

        if self._variable_client:
            self._variable_client.update(wait)


class MADQNRecurrentExecutor(RecurrentExecutor):
    """A feed-forward executor.
    An executor based on a feed-forward policy for each agent in the system
    which takes non-batched observations and outputs non-batched actions.
    It also allows adding experiences to replay and updating the weights
    from the policy on the learner.
    """

    def __init__(
        self,
        q_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        shared_weights: bool = True,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
        trainer: MADQNTrainer = None,
        communication_module: Optional[BaseCommunicationModule] = None,
        evaluator: bool = False,
        fingerprint: bool = False,
    ):
        """[summary]

        Args:
            q_networks (Dict[str, snt.Module]): [description]
            action_selectors (Dict[str, snt.Module]): [description]
            shared_weights (bool, optional): [description]. Defaults to True.
            adder (Optional[adders.ParallelAdder], optional): [description]. Defaults
                to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                [description]. Defaults to None.
            store_recurrent_state (bool, optional): [description]. Defaults to True.
            trainer (MADQNTrainer, optional): [description]. Defaults to None.
            communication_module (Optional[BaseCommunicationModule], optional):
                [description]. Defaults to None.
            evaluator (bool, optional): [description]. Defaults to False.
            fingerprint (bool, optional): [description]. Defaults to False.
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._q_networks = q_networks
        self._policy_networks = q_networks
        self._action_selectors = action_selectors
        self._store_recurrent_state = store_recurrent_state
        self._trainer = trainer
        self._shared_weights = shared_weights

        self._states: Dict[str, Any] = {}

    @tf.function
    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        state: types.NestedTensor,
        legal_actions: types.NestedTensor,
        epsilon: tf.Tensor,
    ) -> types.NestedTensor:
        """[summary]

        Args:
            agent (str): [description]
            observation (types.NestedTensor): [description]
            state (types.NestedTensor): [description]
            legal_actions (types.NestedTensor): [description]
            epsilon (tf.Tensor): [description]

        Returns:
            types.NestedTensor: [description]
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)
        batched_legals = tf2_utils.add_batch_dim(legal_actions)

        # index network either on agent type or on agent id
        agent_key = agent.split("_")[0] if self._shared_weights else agent

        # Compute the policy, conditioned on the observation.
        q_values, new_state = self._q_networks[agent_key](batched_observation, state)

        # select legal action
        action = self._action_selectors[agent_key](
            q_values, batched_legals, epsilon=epsilon
        )

        return action, new_state

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:
        """[summary]

        Args:
            agent (str): [description]
            observation (types.NestedArray): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            types.NestedArray: [description]
        """

        raise NotImplementedError

    def select_actions(
        self, observations: Dict[str, OLT]
    ) -> Dict[str, types.NestedArray]:
        """[summary]

        Args:
            observations (Dict[str, OLT]): [description]

        Returns:
            Dict[str, types.NestedArray]: [description]
        """

        actions = {}

        for agent, observation in observations.items():

            # Pass the observation through the policy network.
            if self._trainer is not None:
                epsilon = self._trainer.get_epsilon()
            else:
                epsilon = 0.0

            epsilon = tf.convert_to_tensor(epsilon)

            policy_output, new_state = self._policy(
                agent,
                observation.observation,
                self._states[agent],
                observation.legal_actions,
                epsilon,
            )

            self._states[agent] = new_state

            actions[agent] = tf2_utils.to_numpy_squeeze(policy_output)

        # Return a numpy array with squeezed out batch dimension.
        return actions


class MADQNRecurrentCommExecutor(RecurrentCommExecutor):
    """A feed-forward executor.
    An executor based on a feed-forward policy for each agent in the system
    which takes non-batched observations and outputs non-batched actions.
    It also allows adding experiences to replay and updating the weights
    from the policy on the learner.
    """

    def __init__(
        self,
        q_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        communication_module: BaseCommunicationModule,
        shared_weights: bool = True,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
        trainer: MADQNTrainer = None,
        fingerprint: bool = False,
        evaluator: bool = False,
    ):
        """[summary]

        Args:
            q_networks (Dict[str, snt.Module]): [description]
            action_selectors (Dict[str, snt.Module]): [description]
            communication_module (BaseCommunicationModule): [description]
            shared_weights (bool, optional): [description]. Defaults to True.
            adder (Optional[adders.ParallelAdder], optional): [description]. Defaults
                to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                [description]. Defaults to None.
            store_recurrent_state (bool, optional): [description]. Defaults to True.
            trainer (MADQNTrainer, optional): [description]. Defaults to None.
            fingerprint (bool, optional): [description]. Defaults to False.
            evaluator (bool, optional): [description]. Defaults to False.
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._q_networks = q_networks
        self._policy_networks = q_networks
        self._communication_module = communication_module
        self._action_selectors = action_selectors
        self._store_recurrent_state = store_recurrent_state
        self._trainer = trainer
        self._shared_weights = shared_weights

        self._states: Dict[str, Any] = {}
        self._messages: Dict[str, Any] = {}

    @tf.function
    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        state: types.NestedTensor,
        message: types.NestedTensor,
        legal_actions: types.NestedTensor,
        epsilon: tf.Tensor,
    ) -> types.NestedTensor:
        """[summary]

        Args:
            agent (str): [description]
            observation (types.NestedTensor): [description]
            state (types.NestedTensor): [description]
            message (types.NestedTensor): [description]
            legal_actions (types.NestedTensor): [description]
            epsilon (tf.Tensor): [description]

        Returns:
            types.NestedTensor: [description]
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)
        batched_legals = tf2_utils.add_batch_dim(legal_actions)

        # index network either on agent type or on agent id
        agent_key = agent.split("_")[0] if self._shared_weights else agent

        # Compute the policy, conditioned on the observation.
        (q_values, m_values), new_state = self._q_networks[agent_key](
            batched_observation, state, message
        )

        # select legal action
        action = self._action_selectors[agent_key](
            q_values, batched_legals, epsilon=epsilon
        )

        return (action, m_values), new_state

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:
        """[summary]

        Args:
            agent (str): [description]
            observation (types.NestedArray): [description]

        Raises:
            NotImplementedError: [description]

        Returns:
            types.NestedArray: [description]
        """

        raise NotImplementedError

    def select_actions(
        self, observations: Dict[str, OLT]
    ) -> Dict[str, types.NestedArray]:
        """[summary]

        Args:
            observations (Dict[str, OLT]): [description]

        Returns:
            Dict[str, types.NestedArray]: [description]
        """

        actions = {}

        message_inputs = self._communication_module.process_messages(self._messages)

        for agent, observation in observations.items():

            # Pass the observation through the policy network.
            if self._trainer is not None:
                epsilon = self._trainer.get_epsilon()
            else:
                epsilon = 0.0

            epsilon = tf.convert_to_tensor(epsilon)

            (policy_output, new_message), new_state = self._policy(
                agent,
                observation.observation,
                self._states[agent],
                message_inputs[agent],
                observation.legal_actions,
                epsilon,
            )

            self._states[agent] = new_state
            self._messages[agent] = new_message

            actions[agent] = tf2_utils.to_numpy_squeeze(policy_output)

        # Return a numpy array with squeezed out batch dimension.
        return actions
