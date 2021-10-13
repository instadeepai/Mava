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
from typing import Any, Dict, List, Optional, Tuple, Union

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme import types
from acme.specs import EnvironmentSpec

# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils
from dm_env import specs

from mava import adders
from mava.systems.tf import executors
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num
from mava.components.tf.modules.exploration.exploration_scheduling import BaseExplorationScheduler

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray
tfd = tfp.distributions


class MADQNScalingFeedForwardExecutor(executors.FeedForwardExecutor):
    """A feed-forward executor for discrete actions.

    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        network_sampling_setup: List,
        net_keys_to_ids: Dict[str, int],
        adder: Optional[adders.ParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        exploration_scheduler: Optional[BaseExplorationScheduler] = None
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

        """

        # Store these for later use.
        self._agent_specs = agent_specs
        self._network_sampling_setup = network_sampling_setup
        self._counts = counts
        self._network_int_keys_extras: Dict[str, np.array] = {}
        self._net_keys_to_ids = net_keys_to_ids
        self._action_selectors = action_selectors
        self._exploration_scheduler = exploration_scheduler

        super().__init__(
            policy_networks=policy_networks,
            agent_net_keys=agent_net_keys,
            adder=adder,
            variable_client=variable_client,
        )

    @tf.function
    def _policy(
        self, 
        agent: str, 
        observation: types.NestedTensor,
        legal_actions: types.NestedTensor,
        epsilon: tf.Tensor,
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
        batched_legals = tf2_utils.add_batch_dim(legal_actions)

        # index network either on agent type or on agent id
        agent_key = self._agent_net_keys[agent]

        # Compute the policy, conditioned on the observation.
        policy = self._policy_networks[agent_key](batched_observation)

        # Select legal action
        action = self._action_selectors[agent_key](
            policy, batched_legals, epsilon=epsilon
        )

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
        # TODO Epsilon scheduling
        if self._exploration_scheduler:
            epsilon = self._exploration_scheduler.get_epsilon()
            self._exploration_scheduler.decrement_epsilon()
        else:
            epsilon = 0.0

        epsilon = tf.convert_to_tensor(epsilon)

        # Step the recurrent policy/value network forward
        # given the current observation and state.
        action, policy = self._policy(
            agent, 
            observation.observation, 
            observation.legal_actions,
            epsilon
        )

        # Return a numpy array with squeezed out batch dimension.
        action = tf2_utils.to_numpy_squeeze(action)
        policy = tf2_utils.to_numpy_squeeze(policy)
        return action, policy

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

        actions = {}
        policies = {}
        for agent, observation in observations.items():
            actions[agent], policies[agent] = self.select_action(agent, observation)
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

        # Select new networks from the sampler at the start of each episode.
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

        actions, _ = actions

        next_extras["network_int_keys"] = self._network_int_keys_extras
        # TODO (dries): Sort out this mypy issue.
        self._adder.add(actions, next_timestep, next_extras)  # type: ignore

    def update(self, wait: bool = False) -> None:
        """Update the policy variables."""
        if self._variable_client:
            self._variable_client.get_async()


class MADQNScalingRecurrentExecutor(executors.RecurrentExecutor):
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
        adder: Optional[adders.ParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
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
        """

        # Store these for later use.
        self._agent_specs = agent_specs
        self._network_sampling_setup = network_sampling_setup
        self._counts = counts
        self._net_keys_to_ids = net_keys_to_ids
        self._network_int_keys_extras: Dict[str, np.array] = {}
        super().__init__(
            policy_networks=policy_networks,
            agent_net_keys=agent_net_keys,
            adder=adder,
            variable_client=variable_client,
            store_recurrent_state=store_recurrent_state,
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

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:
        """select an action for a single agent in the system
        Args:
            agent: agent id
            observation: observation tensor received from the
                environment.
        Returns:
            action and policy.
        """

        # TODO Mask actions here using observation.legal_actions
        # Initialize the RNN state if necessary.
        if self._states[agent] is None:
            # index network either on agent type or on agent id
            agent_key = self._agent_net_keys[agent]
            self._states[agent] = self._policy_networks[agent_key].initia_state(1)

        # Step the recurrent policy forward given the current observation and state.
        action, policy, new_state = self._policy(
            agent, observation.observation, self._states[agent]
        )

        # Bookkeeping of recurrent states for the observe method.
        self._update_state(agent, new_state)

        # Return a numpy array with squeezed out batch dimension.
        action = tf2_utils.to_numpy_squeeze(action)
        policy = tf2_utils.to_numpy_squeeze(policy)
        return action, policy

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

        actions = {}
        policies = {}
        for agent, observation in observations.items():
            actions[agent], policies[agent] = self.select_action(agent, observation)
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
