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

import copy
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
from numpy.random import randint

from mava import adders
from mava.systems.tf import executors

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
        do_pbt: bool = False,
        pbt_samples: List = [],
        adder: Optional[adders.ParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):

        """Initialise the system executor
        Args:
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            agent_specs (Dict[str, EnvironmentSpec]): agent observation and action
                space specifications.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
        """

        # Store these for later use.
        self._agent_specs = agent_specs
        self._do_pbt = do_pbt
        self._pbt_samples = pbt_samples
        self._counts = counts
        self._network_keys_extras: Dict[str, np.array] = {}
        super().__init__(
            policy_networks=policy_networks,
            agent_net_keys=agent_net_keys,
            adder=adder,
            variable_client=variable_client,
        )

    @tf.function
    def _policy(
        self, agent: str, observation: types.NestedTensor
    ) -> types.NestedTensor:
        """Agent specific policy function

        Args:
            agent (str): agent id
            observation (types.NestedTensor): observation tensor received from the
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
            agent (str): agent id.
            observation (types.NestedArray): observation tensor received from the
                environment.

        Returns:
            Tuple[types.NestedArray, types.NestedArray]: agent action and policy.
        """
        # Step the recurrent policy/value network forward
        # given the current observation and state.
        action, policy = self._policy(agent, observation.observation)

        # Return a numpy array with squeezed out batch dimension.
        action = tf2_utils.to_numpy_squeeze(action)
        policy = tf2_utils.to_numpy_squeeze(policy)
        return action, policy

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
        policies = {}
        for agent, observation in observations.items():
            actions[agent], policies[agent] = self.select_action(agent, observation)
        return actions, policies

    def sample_new_agent_keys(self) -> Dict[str, np.array]:
        save_net_keys = {}
        agent_slots = copy.copy(sorted(list(self._agent_net_keys.keys())))
        self._agent_net_keys = {}
        while len(agent_slots) > 0:
            sample = self._pbt_samples[randint(len(self._pbt_samples))]
            for net_key in sample:
                agent = agent_slots.pop(0)
                self._agent_net_keys[agent] = net_key
                save_net_keys[agent] = np.array(net_key, dtype=np.dtype("U10"))
        return save_net_keys

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
            if self._do_pbt:
                """In population based trianing select new networks from the sampler
                at the start of each episode. Also add the network key used by each
                agent."""
                self._network_keys_extras = self.sample_new_agent_keys()
                extras["network_keys"] = self._network_keys_extras

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
            actions (Union[ Dict[str, types.NestedArray], List[Dict[str,
                types.NestedArray]] ]): system agents' actions.
            next_timestep (dm_env.TimeStep): data emitted by an environment during
                interaction.
            next_extras (Dict[str, types.NestedArray], optional): possible extra
                information to record during the transition. Defaults to {}.
        """
        if self._adder:
            _, policy = actions
            if self._do_pbt:
                """In population based trianing select new networks from the sampler
                at the start of each episode. Also add the network key used by each
                agent."""
                next_extras["network_keys"] = self._network_keys_extras
            # TODO (dries): Sort out this mypy issue.
            self._adder.add(policy, next_timestep, next_extras)  # type: ignore

    def update(self, wait: bool = False) -> None:
        """Update the policy variables."""
        if self._variable_client:
            # TODO (dries): Maybe changes this to a async get?
            self._variable_client.get_and_wait()


class MADDPGRecurrentExecutor(executors.RecurrentExecutor):
    """A recurrent executor for MADDPG.
    An executor based on a recurrent policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        agent_net_keys: Dict[str, str],
        adder: Optional[adders.ParallelAdder] = None,
        counts: Optional[Dict[str, Any]] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        store_recurrent_state: bool = True,
    ):
        """Initialise the system executor
        Args:
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            agent_specs (Dict[str, EnvironmentSpec]): agent observation and action
                space specifications.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            counts: (Dict[str, Any], Optional) Used to store the executor step counts.
            Defaults to None.
            store_recurrent_state (bool, optional): boolean to store the recurrent
                network hidden state. Defaults to True.
        """

        # Store these for later use.
        self._agent_specs = agent_specs
        self._counts = counts
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
            observations (Dict[str, types.NestedArray]): agent observations from the
                environment.
        Returns:
            Tuple[Dict[str, types.NestedArray], Dict[str, types.NestedArray]]:
                actions and policies for all agents in the system.
        """

        actions = {}
        policies = {}
        for agent, observation in observations.items():
            actions[agent], policies[agent] = self.select_action(agent, observation)
        return actions, policies

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
        _, policy = actions
        if not self._store_recurrent_state:
            if next_extras:
                # TODO (dries): Sort out this mypy issue.
                self._adder.add(policy, next_timestep, next_extras)  # type: ignore
            else:
                self._adder.add(policy, next_timestep)  # type: ignore
            return

        numpy_states = {
            agent: tf2_utils.to_numpy_squeeze(_state)
            for agent, _state in self._states.items()
        }
        if next_extras:
            next_extras.update({"core_states": numpy_states})
            self._adder.add(policy, next_timestep, next_extras)  # type: ignore
        else:
            self._adder.add(policy, next_timestep, numpy_states)  # type: ignore

    def update(self, wait: bool = False) -> None:
        """Update the policy variables."""
        if self._variable_client:
            # TODO (dries): Maybe changes this to a async get?
            self._variable_client.get_and_wait()
