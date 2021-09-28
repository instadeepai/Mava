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

"""MAPPO system executor implementation."""

from typing import Any, Dict, Optional, Tuple, Union

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme import types
from acme.specs import EnvironmentSpec
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders, core
from mava.systems.tf import executors

tfd = tfp.distributions


class MAPPOFeedForwardExecutor(core.Executor):
    """A feed-forward executor.
    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):
        """Initialise the system executor

        Args:
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
        """
        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._policy_networks = policy_networks
        self._agent_net_keys = agent_net_keys
        self._prev_log_probs: Dict[str, Any] = {}

    @tf.function
    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
    ) -> Tuple[types.NestedTensor, types.NestedTensor]:
        """Agent specific policy function

        Args:
            agent (str): agent id
            observation (types.NestedTensor): observation tensor received from the
                environment.

        Returns:
            Tuple[types.NestedTensor, types.NestedTensor]: log probabilities and action
        """

        # Index network either on agent type or on agent id.
        network_key = self._agent_net_keys[agent]

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        observation = tf2_utils.add_batch_dim(observation.observation)

        # Compute the policy, conditioned on the observation.
        policy = self._policy_networks[network_key](observation)

        # Sample from the policy and compute the log likelihood.
        action = policy.sample()
        log_prob = policy.log_prob(action)

        # Cast for compatibility with reverb.
        # sample() returns a 'int32', which is a problem.
        if isinstance(policy, tfp.distributions.Categorical):
            action = tf.cast(action, "int64")

        return log_prob, action

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:
        """select an action for a single agent in the system

        Args:
            agent (str): agent id.
            observation (types.NestedArray): observation tensor received from the
                environment.

        Returns:
            types.NestedArray: agent action
        """

        # Step the recurrent policy/value network forward
        # given the current observation and state.
        self._prev_log_probs[agent], action = self._policy(agent, observation)

        # Return a numpy array with squeezed out batch dimension.
        action = tf2_utils.to_numpy_squeeze(action)

        # TODO(Kale-ab) : Remove. This is for debugging.
        if np.isnan(action).any():
            print(
                f"Value error- Log Probs:{self._prev_log_probs[agent]} Action: {action} "  # noqa: E501
            )

        return action

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Dict[str, types.NestedArray]:
        """select the actions for all agents in the system

        Args:
           observations (Dict[str, types.NestedArray]): agent observations from the
                environment.

        Returns:
            Dict[str, types.NestedArray]: actions and policies for all agents in
                the system.
        """

        actions = {}
        for agent, observation in observations.items():
            action = self.select_action(agent, observation)
            actions[agent] = action

        return actions

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
        if self._adder:
            self._adder.add_first(timestep)

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

        raise NotImplementedError(
            "Need to fix the FeedForward case to reflect the recurrent updates."
        )

        if not self._adder:
            return

        next_extras.update({"log_probs": self._prev_log_probs})

        next_extras = tf2_utils.to_numpy_squeeze(next_extras)

        self._adder.add(actions, next_timestep, next_extras)

    def update(self, wait: bool = False) -> None:
        """update executor variables

        Args:
            wait (bool, optional): whether to stall the executor's request for new
                variables. Defaults to False.
        """

        if self._variable_client:
            self._variable_client.update(wait)


class MAPPORecurrentExecutor(executors.RecurrentExecutor):
    """A recurrent executor for MADDPG.
    An executor based on a recurrent policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):
        """Initialise the system executor
        Args:
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
        """
        super().__init__(
            policy_networks=policy_networks,
            agent_net_keys=agent_net_keys,
            adder=adder,
            variable_client=variable_client,
            store_recurrent_state=True,
        )

    @tf.function
    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        state: types.NestedTensor,
    ) -> Tuple[types.NestedTensor, types.NestedTensor, types.NestedTensor]:
        """Agent specific policy function

        Args:
            agent (str): agent id
            observation (types.NestedTensor): observation tensor received from the
                environment.
            state (types.NestedTensor): recurrent network state.

        Raises:
            NotImplementedError: unknown action space

        Returns:
            Tuple[types.NestedTensor, types.NestedTensor, types.NestedTensor]:
                action, policy and new recurrent hidden state
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # index network either on agent type or on agent id
        agent_key = self._agent_net_keys[agent]

        # Compute the policy, conditioned on the observation.
        logits, new_state = self._policy_networks[agent_key](batched_observation, state)

        # Sample from the policy and compute the log likelihood.
        action = tfd.Categorical(logits).sample()
        # action = tf2_utils.to_numpy_squeeze(action)

        # Cast for compatibility with reverb.
        # sample() returns a 'int32', which is a problem.
        # if isinstance(policy, tfp.distributions.Categorical):
        action = tf.cast(action, "int64")

        return action, logits, new_state

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:
        """select an action for a single agent in the system

        Args:
            agent (str): agent id
            observation (types.NestedArray): observation tensor received from the
                environment.

        Returns:
            types.NestedArray: action and policy.
        """

        # TODO Mask actions here using observation.legal_actions
        # Initialize the RNN state if necessary.
        if self._states[agent] is None:
            # index network either on agent type or on agent id
            agent_key = self._agent_net_keys[agent]
            self._states[agent] = self._policy_networks[agent_key].initia_state(1)

        # Step the recurrent policy forward given the current observation and state.
        action, log_prob, new_state = self._policy(
            agent, observation.observation, self._states[agent]
        )

        # Bookkeeping of recurrent states for the observe method.
        self._update_state(agent, new_state)

        # Return a numpy array with squeezed out batch dimension.
        action = tf2_utils.to_numpy_squeeze(action)
        log_prob = tf2_utils.to_numpy_squeeze(log_prob)
        return action, log_prob

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]]:
        """select the actions for all agents in the system

        Args:
            observations (Dict[str, types.NestedArray]): agent observations from the
                environment.

        Returns:
            Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]]:
                actions and policies for all agents in the system.
        """

        actions = {}
        log_probs = {}
        for agent, observation in observations.items():
            actions[agent], log_probs[agent] = self.select_action(agent, observation)
        return actions, log_probs

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Optional[Dict[str, types.NestedArray]] = {},
    ) -> None:
        """record observed timestep from the environment

        Args:
            actions (Dict[str, types.NestedArray]): system agents' actions.
            next_timestep (dm_env.TimeStep): data emitted by an environment during
                interaction.
            next_extras (Dict[str, types.NestedArray], optional): possible extra
                information to record during the transition. Defaults to {}.
        """

        if not self._adder:
            return
        actions, log_probs = actions

        adder_actions = {}
        for agent in actions.keys():
            adder_actions[agent] = {}
            adder_actions[agent]["actions"] = actions[agent]
            adder_actions[agent]["log_probs"] = log_probs[agent]

        numpy_states = {
            agent: tf2_utils.to_numpy_squeeze(_state)
            for agent, _state in self._states.items()
        }
        if next_extras:
            next_extras.update({"core_states": numpy_states})
            self._adder.add(adder_actions, next_timestep, next_extras)  # type: ignore
        else:
            self._adder.add(adder_actions, next_timestep, numpy_states)  # type: ignore
