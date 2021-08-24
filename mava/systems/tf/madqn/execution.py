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
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.components.tf.modules.stabilising.fingerprints import FingerPrintStabalisation
from mava.systems.tf.executors import (
    FeedForwardExecutor,
    RecurrentCommExecutor,
    RecurrentExecutor,
)
from mava.systems.tf.madqn.training import MADQNTrainer
from mava.types import OLT


class MADQNFeedForwardExecutor(FeedForwardExecutor):
    """A feed-forward executor.
    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        q_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        trainer: MADQNTrainer,
        agent_net_keys: Dict[str, str],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        fingerprint_module: FingerPrintStabalisation = None,
        evaluator: bool = False,
    ):
        """Initialise the system executor

        Args:
            q_networks (Dict[str, snt.Module]): q-value networks for each agent in the
                system.
            action_selectors (Dict[str, Any]): policy action selector method, e.g.
                epsilon greedy.
            trainer (MADQNTrainer, optional): system trainer.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            communication_module (BaseCommunicationModule): module for enabling
                communication protocols between agents. Defaults to None.
            fingerprint (bool, optional): whether to use fingerprint stabilisation to
                stabilise experience replay. Defaults to False.
            evaluator (bool, optional): whether the executor will be used for
                evaluation. Defaults to False.
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._q_networks = q_networks
        self._action_selectors = action_selectors
        self._trainer = trainer
        self._agent_net_keys = agent_net_keys
        self._fingerprint_module = fingerprint_module
        self._evaluator = evaluator

    @tf.function
    def _policy(
        self,
        q_network: snt.Module,
        action_selector: Any,
        observation: types.NestedTensor,
        legal_actions: types.NestedTensor,
        epsilon: tf.Tensor,
    ) -> types.NestedTensor:
        """Agent specific policy function

        Args:
            agent (str): agent id
            observation (types.NestedTensor): observation tensor received from the
                environment.
            legal_actions (types.NestedTensor): actions allowed to be taken at the
                current observation.
            epsilon (tf.Tensor): value for epsilon greedy action selection.
            fingerprint (Optional[tf.Tensor], optional): policy fingerprints. Defaults
                to None.

        Returns:
            types.NestedTensor: agent action
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)
        batched_legals = tf2_utils.add_batch_dim(legal_actions)

        q_values = q_network(batched_observation)

        # Select legal action.
        action = action_selector(
            q_values, batched_legals, epsilon=epsilon
        )

        return action

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:
        """select an action for a single agent in the system

        Args:
            agent (str): agent id
            observation (types.NestedArray): observation tensor received from the
                environment.

        Returns:
            types.NestedArray: agent action
        """
        # Set epsilon depending
        if self._evaluator:
            # Near zero.
            # Zero causes an error sometimes.
            epsilon = 1e-10
        else:
            epsilon = self._trainer.get_epsilon()
        # Convert to tensor.
        epsilon = tf.convert_to_tensor(epsilon, dtype="float32")

        # Get legal actions and observations.
        legal_actions = tf.convert_to_tensor(observation.legal_actions)
        observation_tensor = tf.convert_to_tensor(observation.observation)

        # Maybe do fingerprinting.
        if self._fingerprint_module is not None:
            info = {
                "trainer_step": self._trainer.get_trainer_steps(),
                "epsilon": self._trainer.get_epsilon(),
            }
            # Apply fingerprinting hook.
            observation_tensor = self._fingerprint_module.executor_act_hook(
                observation_tensor, info
            )

        # index network either on agent type or on agent id
        agent_net_key = self._agent_net_keys[agent]

        # Get q_network and action selector.
        q_network = self._q_networks[agent_net_key]
        action_selector = self._action_selectors[agent_net_key]

        # Apply epsilon-greedy policy
        action = self._policy(
            q_network,
            action_selector,
            observation_tensor,
            legal_actions,
            epsilon,
        )

        # Squeeze out batch dimension.
        action = tf2_utils.to_numpy_squeeze(action)

        return action

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """record first observed timestep from the environment

        Args:
            timestep (dm_env.TimeStep): data emitted by an environment at first step of
                interaction.
            extras (Dict[str, types.NestedArray], optional): possible extra information
                to record during the first step. Defaults to {}.
        """

        # Maybe apply fingerprinting.
        if self._fingerprint_module is not None:
            # Get useful info for fingerprinting
            info = {
                "trainer_step": self._trainer.get_trainer_steps(),
                "epsilon": self._trainer.get_epsilon(),
            }
            extras = self._fingerprint_module.executor_observe_hook(extras, info=info)

        if self._adder:
            self._adder.add_first(timestep, extras)

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """record observed timestep from the environment

        Args:
            actions (Dict[str, types.NestedArray]): system agents' actions.
            next_timestep (dm_env.TimeStep): data emitted by an environment during
                interaction.
            next_extras (Dict[str, types.NestedArray], optional): possible extra
                information to record during the transition. Defaults to {}.
        """
        # Maybe apply fingerprinting.
        if self._fingerprint_module is not None:
            # Get useful info for fingerprinting
            info = {
                "trainer_step": self._trainer.get_trainer_steps(),
                "epsilon": self._trainer.get_epsilon(),
            }
            next_extras = self._fingerprint_module.executor_observe_hook(
                next_extras, info=info
            )

        if self._adder:
            self._adder.add(actions, next_timestep, next_extras)

    def select_actions(
        self, observations: Dict[str, OLT]
    ) -> Dict[str, types.NestedArray]:
        """select the actions for all agents in the system

        Args:
            observations (Dict[str, OLT]): transition object containing observations,
                legal actions and terminals.

        Returns:
            Dict[str, types.NestedArray]: actions for all agents in the system.
        """

        actions = {}
        for agent, observation in observations.items():
            # Select action for agent.
            actions[agent] = self.select_action(agent, observation)

        return actions

    def update(self, wait: bool = False) -> None:
        """update executor variables

        Args:
            wait (bool, optional): whether to stall the executor's request for new
                variables. Defaults to False.
        """

        if self._variable_client:
            self._variable_client.update(wait)


class MADQNRecurrentExecutor(RecurrentExecutor):
    """A recurrent executor.
    An executor based on a recurrent policy for each agent in the system
    """

    def __init__(
        self,
        q_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
        trainer: MADQNTrainer = None,
        fingerprint: bool = False,
        evaluator: bool = False,
    ):
        """Initialise the system executor

        Args:
            q_networks (Dict[str, snt.Module]): q-value networks for each agent in the
                system.
            action_selectors (Dict[str, Any]): policy action selector method, e.g.
                epsilon greedy.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            agent_net_keys (Dict[str, Any]): specifies what network each agent uses.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            store_recurrent_state (bool, optional): boolean to store the recurrent
                network hidden state. Defaults to True.
            trainer (MADQNTrainer, optional): system trainer. Defaults to None.
            communication_module (BaseCommunicationModule): module for enabling
                communication protocols between agents. Defaults to None.
            fingerprint (bool, optional): whether to use fingerprint stabilisation to
                stabilise experience replay. Defaults to False.
            evaluator (bool, optional): whether the executor will be used for
                evaluation. Defaults to False.
        """

        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._q_networks = q_networks
        self._policy_networks = q_networks
        self._action_selectors = action_selectors
        self._store_recurrent_state = store_recurrent_state
        self._trainer = trainer
        self._agent_net_keys = agent_net_keys

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
        """Agent specific policy function

        Args:
            agent (str): agent id
            observation (types.NestedTensor): observation tensor received from the
                environment.
            state (types.NestedTensor): recurrent network state.
            message (types.NestedTensor): received agent messsage.
            legal_actions (types.NestedTensor): actions allowed to be taken at the
                current observation.
            epsilon (tf.Tensor): value for epsilon greedy action selection.

        Returns:
            types.NestedTensor: action and new recurrent hidden state
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)
        batched_legals = tf2_utils.add_batch_dim(legal_actions)

        # index network either on agent type or on agent id
        agent_key = self._agent_net_keys[agent]

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
        """select an action for a single agent in the system

        Args:
            agent (str): agent id
            observation (types.NestedArray): observation tensor received from the
                environment.

        Raises:
            NotImplementedError: has not been implemented for this training type.
        """

        raise NotImplementedError

    def select_actions(
        self, observations: Dict[str, OLT]
    ) -> Dict[str, types.NestedArray]:
        """select the actions for all agents in the system

        Args:
            observations (Dict[str, OLT]): transition object containing observations,
                legal actions and terminals.

        Returns:
            Dict[str, types.NestedArray]: actions for all agents in the system.
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
    """A recurrent executor with communication.
    An executor based on a recurrent policy for each agent in the system using learned
    communication.
    """

    def __init__(
        self,
        q_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        communication_module: BaseCommunicationModule,
        agent_net_keys: Dict[str, str],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
        trainer: MADQNTrainer = None,
        fingerprint: bool = False,
        evaluator: bool = False,
    ):
        """Initialise the system executor

        Args:
            q_networks (Dict[str, snt.Module]): q-value networks for each agent in the
                system.
            action_selectors (Dict[str, Any]): policy action selector method, e.g.
                epsilon greedy.
            communication_module (BaseCommunicationModule): module for enabling
                communication protocols between agents.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            store_recurrent_state (bool, optional): boolean to store the recurrent
                network hidden state. Defaults to True.
            trainer (MADQNTrainer, optional): system trainer. Defaults to None.
            fingerprint (bool, optional): whether to use fingerprint stabilisation to
                stabilise experience replay. Defaults to False.
            evaluator (bool, optional): whether the executor will be used for
                evaluation. Defaults to False.
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
        self._agent_net_keys = agent_net_keys

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
        """Agent specific policy function

        Args:
            agent (str): agent id
            observation (types.NestedTensor): observation tensor received from the
                environment.
            state (types.NestedTensor): Recurrent network state.
            message (types.NestedTensor): received agent messsage.
            legal_actions (types.NestedTensor): actions allowed to be taken at the
                current observation.
            epsilon (tf.Tensor): value for epsilon greedy action selection.

        Returns:
            types.NestedTensor: action and new recurrent hidden state
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)
        batched_legals = tf2_utils.add_batch_dim(legal_actions)

        # index network either on agent type or on agent id
        agent_key = self._agent_net_keys[agent]

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
        """select an action for a single agent in the system

        Args:
            agent (str): agent id
            observation (types.NestedArray): observation tensor received from the
                environment.

        Raises:
            NotImplementedError: has not been implemented for this training type.
        """

        raise NotImplementedError

    def select_actions(
        self, observations: Dict[str, OLT]
    ) -> Dict[str, types.NestedArray]:
        """select the actions for all agents in the system

        Args:
            observations (Dict[str, OLT]): transition object containing observations,
                legal actions and terminals.

        Returns:
            Dict[str, types.NestedArray]: actions for all agents in the system.
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
