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

"""MADDPG system executor implementation."""
from typing import Any, Dict, List, Optional, Tuple, Union

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
from acme import types
from acme.specs import EnvironmentSpec

# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from dm_env import specs

from mava import adders
from mava.systems.tf import executors
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray
tfd = tfp.distributions


class MADDPGFeedForwardExecutor(executors.FeedForwardExecutor):
    """A feed-forward executor for discrete actions.
    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        network_sampling_setup: List,
        net_keys_to_ids: Dict[str, int],
        evaluator: bool = False,
        adder: Optional[adders.ReverbParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        interval: Optional[dict] = None,
    ):

        """Initialise the system executor
        Args:
            policy_networks: policy networks for each agent in
                the system.
            agent_specs: agent observation and action
                space specifications.
            agent_net_keys: specifies what network each agent uses.
            network_sampling_setup: List of networks that are randomly
                sampled from by the executors at the start of an environment run.
            net_keys_to_ids: Specifies a mapping from network keys to their integer id.
            adder: adder which sends data
                to a replay buffer. Defaults to None.
            counts: Count values used to record excutor episode and steps.
            variable_client:
                client to copy weights from the trainer. Defaults to None.
            evaluator: whether the executor will be used for
                evaluation.
            interval: interval that evaluations are run at.
        """

        # Store these for later use.
        self._agent_specs = agent_specs
        self._network_sampling_setup = network_sampling_setup
        self._counts = counts
        self._network_int_keys_extras: Dict[str, np.ndarray] = {}
        self._net_keys_to_ids = net_keys_to_ids
        self._evaluator = evaluator
        self._interval = interval
        super().__init__(
            policy_networks=policy_networks,
            agent_net_keys=agent_net_keys,
            adder=adder,
            variable_client=variable_client,
        )

    def _policy(
        self, agent: str, observation: types.NestedTensor
    ) -> types.NestedTensor:
        """Agent specific policy function

        Args:
            agent: agent id
            observation: observation tensor received from the
                environment.

        Raises:
            NotImplementedError: unknown action space

        Returns:
            types.NestedTensor: agent action
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # index network either on agent type or on agent id
        agent_key = self._agent_net_keys[agent]

        # Compute the policy, conditioned on the observation.
        policy = self._policy_networks[agent_key](batched_observation)

        # TODO (dries): Make this support hybrid action spaces.
        if type(self._agent_specs[agent].actions) == BoundedArray:
            # Continuous action
            action = policy
        elif type(self._agent_specs[agent].actions) == DiscreteArray:
            action = tf.math.argmax(policy, axis=1)
        else:
            raise NotImplementedError

        return action, policy

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> Tuple[types.NestedArray, types.NestedArray]:
        """select an action for a single agent in the system

        Args:
            agent: agent id.
            observation: observation tensor received
            from the environment.

        Returns:
            agent action and policy.
        """
        # Step the recurrent policy/value network forward
        # given the current observation and state.
        action, policy = self._policy(agent, observation.observation)
        return action, policy

    @tf.function
    def _select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]]:
        """Select the actions for all agents in the system"""
        actions = {}
        policies = {}
        for agent, observation in observations.items():
            actions[agent], policies[agent] = self.select_action(agent, observation)
        return actions, policies

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]]:
        """select the actions for all agents in the system

        Args:
            observations: agent observations from the
                environment.

        Returns:
            actions and policies for all agents in the system.
        """

        actions, policies = self._select_actions(observations)
        actions = tree.map_structure(tf2_utils.to_numpy_squeeze, actions)
        policies = tree.map_structure(tf2_utils.to_numpy_squeeze, policies)
        return actions, policies

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """record first observed timestep from the environment
        Args:
            timestep: data emitted by an environment at first step of
                interaction.
            extras: possible extra information
                to record during the first step. Defaults to {}.
        """
        if not self._adder:
            return

        "Select new networks from the sampler at the start of each episode."
        agents = sort_str_num(list(self._agent_net_keys.keys()))
        self._network_int_keys_extras, self._agent_net_keys = sample_new_agent_keys(
            agents,
            self._network_sampling_setup,
            self._net_keys_to_ids,
        )

        extras["network_int_keys"] = self._network_int_keys_extras
        self._adder.add_first(timestep, extras)

    def observe(
        self,
        actions: Union[
            Dict[str, types.NestedArray], List[Dict[str, types.NestedArray]]
        ],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """record observed timestep from the environment
        Args:
            actions: system agents' actions.
            next_timestep: data emitted by an environment during
                interaction.
            next_extras: possible extra
                information to record during the transition. Defaults to {}.
        """
        if not self._adder:
            return
        _, policy = actions
        next_extras["network_int_keys"] = self._network_int_keys_extras
        # TODO (dries): Sort out this mypy issue.
        self._adder.add(policy, next_timestep, next_extras)  # type: ignore

    def update(self, wait: bool = False) -> None:
        """Update the policy variables."""
        if self._variable_client:
            self._variable_client.get_async()


class MADDPGRecurrentExecutor(executors.RecurrentExecutor):
    """A recurrent executor for MADDPG.
    An executor based on a recurrent policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        network_sampling_setup: List,
        net_keys_to_ids: Dict[str, int],
        evaluator: bool = False,
        adder: Optional[adders.ReverbParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
        interval: Optional[dict] = None,
    ):
        """Initialise the system executor
        Args:
            policy_networks: policy networks for each agent in
                the system.
            agent_specs: agent observation and action
                space specifications.
            agent_net_keys: specifies what network each agent uses.
            network_sampling_setup: List of networks that are randomly
                sampled from by the executors at the start of an environment run.
            net_keys_to_ids: Specifies a mapping from network keys to their integer id.
            adder: adder which sends data
                to a replay buffer. Defaults to None.
            counts: Count values used to record excutor episode and steps.
            variable_client:
                client to copy weights from the trainer. Defaults to None.
            store_recurrent_state: boolean to store the recurrent
                network hidden state. Defaults to True.
            evaluator: whether the executor will be used for
                evaluation.
            interval: interval that evaluations are run at.
        """

        # Store these for later use.
        self._agent_specs = agent_specs
        self._network_sampling_setup = network_sampling_setup
        self._counts = counts
        self._net_keys_to_ids = net_keys_to_ids
        self._network_int_keys_extras: Dict[str, np.ndarray] = {}
        self._evaluator = evaluator
        self._interval = interval
        super().__init__(
            policy_networks=policy_networks,
            agent_net_keys=agent_net_keys,
            adder=adder,
            variable_client=variable_client,
            store_recurrent_state=store_recurrent_state,
        )

    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        state: types.NestedTensor,
    ) -> Tuple[types.NestedTensor, types.NestedTensor, types.NestedTensor]:
        """Agent specific policy function
        Args:
            agent: agent id
            observation: observation tensor received from the
                environment.
            state: recurrent network state.
        Raises:
            NotImplementedError: unknown action space
        Returns:
            action, policy and new recurrent hidden state
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # index network either on agent type or on agent id
        agent_key = self._agent_net_keys[agent]

        # Compute the policy, conditioned on the observation.
        policy, new_state = self._policy_networks[agent_key](batched_observation, state)

        # TODO (dries): Make this support hybrid action spaces.
        if type(self._agent_specs[agent].actions) == BoundedArray:
            # Continuous action
            action = policy
        elif type(self._agent_specs[agent].actions) == DiscreteArray:
            action = tf.math.argmax(policy, axis=1)
        else:
            raise NotImplementedError
        return action, policy, new_state

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]]:
        """select the actions for all agents in the system

        Args:
            observations: agent observations from the
                environment.

        Returns:
            actions and policies for all agents in the system.
        """

        actions, policies, self._states = self._select_actions(
            observations, self._states
        )
        actions = tree.map_structure(tf2_utils.to_numpy_squeeze, actions)
        policies = tree.map_structure(tf2_utils.to_numpy_squeeze, policies)
        return actions, policies

    @tf.function
    def _select_actions(
        self,
        observations: Dict[str, types.NestedArray],
        states: Dict[str, types.NestedArray],
    ) -> Tuple[
        Dict[str, types.NestedArray],
        Dict[str, types.NestedArray],
        Dict[str, types.NestedArray],
    ]:
        """select the actions for all agents in the system
        Args:
            observations: agent observations from the
                environment.
        Returns:
            actions and policies for all agents in the system.
        """

        actions = {}
        policies = {}
        new_states = {}
        for agent, observation in observations.items():
            actions[agent], policies[agent], new_states[agent] = self.select_action(
                agent, observation, states[agent]
            )
        return actions, policies, new_states

    def select_action(  # type: ignore
        self,
        agent: str,
        observation: types.NestedArray,
        agent_state: types.NestedArray,
    ) -> Tuple[types.NestedArray, types.NestedArray, types.NestedArray]:
        """select an action for a single agent in the system
        Args:
            agent: agent id
            observation: observation tensor received from the
                environment.
        Returns:
            action and policy.
        """
        # Step the recurrent policy forward given the current observation and state.
        action, policy, new_state = self._policy(
            agent, observation.observation, agent_state
        )
        return action, policy, new_state

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """record first observed timestep from the environment

        Args:
            timestep: data emitted by an environment at first step of
                interaction.
            extras: possible extra information
                to record during the first step.
        """

        # Re-initialize the RNN state.
        for agent, _ in timestep.observation.items():
            # index network either on agent type or on agent id
            agent_key = self._agent_net_keys[agent]
            self._states[agent] = self._policy_networks[agent_key].initial_state(1)

        if not self._adder:
            return

        # Sample new agent_net_keys.
        agents = sort_str_num(list(self._agent_net_keys.keys()))
        self._network_int_keys_extras, self._agent_net_keys = sample_new_agent_keys(
            agents,
            self._network_sampling_setup,
            self._net_keys_to_ids,
        )

        if self._store_recurrent_state:
            numpy_states = {
                agent: tf2_utils.to_numpy_squeeze(_state)
                for agent, _state in self._states.items()
            }
            extras.update({"core_states": numpy_states})
        extras["network_int_keys"] = self._network_int_keys_extras
        self._adder.add_first(timestep, extras)

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """record observed timestep from the environment
        Args:
            actions: system agents' actions.
            next_timestep: data emitted by an environment during
                interaction.
            next_extras: possible extra
                information to record during the transition.
        """

        if not self._adder:
            return
        _, policy = actions

        if self._store_recurrent_state:
            numpy_states = {
                agent: tf2_utils.to_numpy_squeeze(_state)
                for agent, _state in self._states.items()
            }
            next_extras.update({"core_states": numpy_states})
        next_extras["network_int_keys"] = self._network_int_keys_extras
        self._adder.add(policy, next_timestep, next_extras)  # type: ignore

    def update(self, wait: bool = False) -> None:
        """Update the policy variables."""
        if self._variable_client:
            self._variable_client.get_async()
