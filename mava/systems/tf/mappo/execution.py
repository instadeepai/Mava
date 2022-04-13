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

from typing import Any, Dict, List, Optional, Tuple

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
from acme import types
from acme.specs import EnvironmentSpec
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.systems.tf import executors
from mava.types import OLT
from mava.utils.sort_utils import sample_new_agent_keys, sort_str_num
from mava.utils.training_utils import action_mask_categorical_policies

tfd = tfp.distributions


class MAPPOFeedForwardExecutor(executors.FeedForwardExecutor):
    """A feed-forward executor.

    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        network_sampling_setup: List,
        fix_sampler: Optional[List],
        net_keys_to_ids: Dict[str, int],
        adder: Optional[adders.ReverbParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        evaluator: bool = False,
        interval: Optional[dict] = None,
    ):
        """Initialise the system executor
        Args:
            policy_networks: policy networks for each agent in
                the system.
            adder: adder which sends data to a replay buffer.
            variable_client: client to copy weights from the trainer.
            agent_net_keys: specifies what network each agent uses.
            network_sampling_setup: List of networks that are randomly
                sampled from by the executors at the start of an environment run.
            fix_sampler: Optional list that can fix the executor sampler to sample
                in a specific way.
            net_keys_to_ids: Specifies a mapping from network keys to their integer id.
            evaluator: whether the executor will be used for
                evaluation. Defaults to False.
            interval: interval that evaluations are run at.
        """
        self._agent_specs = agent_specs
        self._network_sampling_setup = network_sampling_setup
        self._fix_sampler = fix_sampler
        self._counts = counts
        self._network_int_keys_extras: Dict[str, Any] = {}
        self._net_keys_to_ids = net_keys_to_ids
        super().__init__(
            policy_networks=policy_networks,
            agent_net_keys=agent_net_keys,
            adder=adder,
            variable_client=variable_client,
        )
        # Store these for later use.
        self._adder = adder
        self._variable_client = variable_client
        self._policy_networks = policy_networks
        self._agent_net_keys = agent_net_keys
        self._prev_log_probs: Dict[str, Any] = {}
        self._interval = interval
        self._evaluator = evaluator

    def _policy(
        self,
        agent: str,
        observation_olt: OLT,
    ) -> Tuple[types.NestedTensor, types.NestedTensor]:
        """Agent specific policy function

        Args:
            agent (str): agent id
            observation (types.NestedTensor): observation tensor received from the
                environment.

        Raises:
            NotImplementedError: unknown action space

        Returns:
            Tuple[types.NestedTensor, types.NestedTensor, types.NestedTensor]:
                action and policy logits
        """

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation_olt.observation)

        # index network either on agent type or on agent id
        agent_key = self._agent_net_keys[agent]

        # Compute the policy, conditioned on the observation.
        policy = self._policy_networks[agent_key](batched_observation)

        # Mask categorical policies using legal actions
        if hasattr(observation_olt, "legal_actions") and isinstance(
            policy, tfp.distributions.Categorical
        ):
            batched_legals = tf2_utils.add_batch_dim(observation_olt.legal_actions)

            policy = action_mask_categorical_policies(
                policy=policy, batched_legal_actions=batched_legals
            )

        # Sample from the policy and compute the log likelihood.
        action = policy.sample()

        log_prob = policy.log_prob(action)
        return action, log_prob

    @tf.function
    def _select_actions(
        self, observations: Dict[str, OLT]
    ) -> Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]]:
        """Select the actions for all agents in the system
        Args:
           observations (Dict[str, types.NestedArray]): agent observations from the
                environment.
        Returns:
            Dict[str, types.NestedArray]: actions and policies for all agents in
                the system.
        """

        actions = {}
        log_probs = {}
        for agent, observation in observations.items():
            # Step the recurrent policy/value network forward
            # given the current observation and state.
            action, log_prob = self._policy(agent, observation)

            # Return a numpy array with squeezed out batch dimension.
            actions[agent] = action
            log_probs[agent] = log_prob

        return actions, log_probs

    def select_actions(self, observations: Dict[str, OLT]) -> types.NestedArray:
        """Select the actions for all agents in the system
        Args:
            observations: agent observations from the
                environment.
        Returns:
            actions.
        """

        actions, log_probs = self._select_actions(observations)
        actions = tree.map_structure(tf2_utils.to_numpy_squeeze, actions)
        log_probs = tree.map_structure(tf2_utils.to_numpy_squeeze, log_probs)
        return actions, log_probs

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
            self._fix_sampler,
        )
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
            actions (Dict[str, types.NestedArray]): system agents' actions.
            next_timestep (dm_env.TimeStep): data emitted by an environment during
                interaction.
            next_extras (Dict[str, types.NestedArray], optional): possible extra
                information to record during the transition. Defaults to {}.
        """

        if not self._adder:
            return
        actions, log_probs = actions  # type: ignore

        adder_actions: Dict[str, Any] = {}
        for agent in actions.keys():
            adder_actions[agent] = {}
            adder_actions[agent]["actions"] = actions[agent]
            adder_actions[agent]["log_probs"] = log_probs[agent]  # type: ignore

        next_extras["network_int_keys"] = self._network_int_keys_extras
        self._adder.add(adder_actions, next_timestep, next_extras)  # type: ignore

    def update(self, wait: bool = False) -> None:
        """Update the policy variables."""
        if self._variable_client:
            self._variable_client.get_async()


class MAPPORecurrentExecutor(executors.RecurrentExecutor):
    """A recurrent executor for MAPPO.
    An executor based on a recurrent policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        network_sampling_setup: List,
        fix_sampler: Optional[List],
        net_keys_to_ids: Dict[str, int],
        adder: Optional[adders.ReverbParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        evaluator: bool = False,
        interval: Optional[dict] = None,
    ):
        """Initialise the system executor
        Args:
            policy_networks: policy networks for each agent in
                the system.
            agent_net_keys: specifies what network each agent uses.
            network_sampling_setup: List of networks that are randomly
                sampled from by the executors at the start of an environment run.
            fix_sampler: Optional list that can fix the executor sampler to sample
                in a specific way.
            net_keys_to_ids: Specifies a mapping from network keys to their integer id.
            adder: adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            evaluator (bool, optional): whether the executor will be used for
                evaluation. Defaults to False.
            interval: interval that evaluations are run at.
        """
        self._agent_specs = agent_specs
        self._network_sampling_setup = network_sampling_setup
        self._fix_sampler = fix_sampler
        self._counts = counts
        self._net_keys_to_ids = net_keys_to_ids
        self._network_int_keys_extras: Dict[str, Any] = {}
        self._evaluator = evaluator
        self._interval = interval
        super().__init__(
            policy_networks=policy_networks,
            agent_net_keys=agent_net_keys,
            adder=adder,
            variable_client=variable_client,
            store_recurrent_state=True,
        )

    def _policy(
        self,
        agent: str,
        observation_olt: types.NestedTensor,
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
        # index network either on agent type or on agent id
        agent_key = self._agent_net_keys[agent]
        batched_observation = tf2_utils.add_batch_dim(observation_olt.observation)

        # Compute the policy, conditioned on the observation.
        policy, new_state = self._policy_networks[agent_key](batched_observation, state)

        # Mask categorical policies using legal actions
        if hasattr(observation_olt, "legal_actions") and isinstance(
            policy, tfp.distributions.Categorical
        ):
            batched_legals = tf2_utils.add_batch_dim(observation_olt.legal_actions)

            policy = action_mask_categorical_policies(
                policy=policy, batched_legal_actions=batched_legals
            )

        # Sample from the policy and compute the log likelihood.
        action = policy.sample()
        log_prob = policy.log_prob(action)
        return action, log_prob, new_state

    def select_actions(self, observations: Dict[str, OLT]) -> types.NestedArray:
        """Select the actions for all agents in the system
        Args:
            observations: agent observations from the
                environment.
        Returns:
            actions.
        """

        actions, log_probs, self._states = self._select_actions(
            observations, self._states
        )
        actions = tree.map_structure(tf2_utils.to_numpy_squeeze, actions)
        log_probs = tree.map_structure(tf2_utils.to_numpy_squeeze, log_probs)
        return actions, log_probs

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
            observations (Dict[str, types.NestedArray]): agent observations from the
                environment.

        Returns:
            Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]]:
                actions and policies for all agents in the system.
        """

        actions = {}
        log_probs = {}
        new_states = {}
        for agent, observation in observations.items():
            actions[agent], log_probs[agent], new_states[agent] = self._policy(
                agent, observation, states[agent]
            )
        return actions, log_probs, new_states

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record first observed timestep from the environment

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
            self._fix_sampler,
        )

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
        """Record observed timestep from the environment

        Args:
            actions (Dict[str, types.NestedArray]): system agents' actions.
            next_timestep (dm_env.TimeStep): data emitted by an environment during
                interaction.
            next_extras (Dict[str, types.NestedArray], optional): possible extra
                information to record during the transition. Defaults to {}.
        """

        if not self._adder:
            return
        actions, log_probs = actions  # type: ignore

        adder_actions: Dict[str, Any] = {}
        for agent in actions.keys():
            adder_actions[agent] = {}
            adder_actions[agent]["actions"] = actions[agent]
            adder_actions[agent]["log_probs"] = log_probs[agent]  # type: ignore

        numpy_states = {
            agent: tf2_utils.to_numpy_squeeze(_state)
            for agent, _state in self._states.items()
        }
        next_extras.update({"core_states": numpy_states})
        next_extras["network_int_keys"] = self._network_int_keys_extras
        self._adder.add(adder_actions, next_timestep, next_extras)  # type: ignore

    def update(self, wait: bool = False) -> None:
        """Update the policy variables."""
        if self._variable_client:
            self._variable_client.get_async()
