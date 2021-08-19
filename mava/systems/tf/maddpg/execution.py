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
        executor_samples: List,
        net_to_ints: Dict[str, int],
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
        self._executor_samples = executor_samples
        self._counts = counts
        self._network_int_keys_extras: Dict[str, np.array] = {}
        self._net_to_ints = net_to_ints
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
            observation (types.NestedArray): observation tensor received
            from the environment.

        Returns:
            Tuple[types.NestedArray, types.NestedArray]: agent action and
            policy.
        """
        # Step the recurrent policy/value network forward
        # given the current observation and state.
        action, policy = self._policy(agent, observation.observation)

        # Return a numpy array with squeezed out batch dimension.

        # TODO (dries): This is always a tensor.
        # Maybe a tree operation is not needed here?
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
        if not self._adder:
            return

        "Select new networks from the sampler at the start of each episode."
        agents = sort_str_num(list(self._agent_net_keys.keys()))
        self._network_int_keys_extras, self._agent_net_keys = sample_new_agent_keys(
            agents,
            self._executor_samples,
            self._net_to_ints,
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
            actions (Union[ Dict[str, types.NestedArray], List[Dict[str,
                types.NestedArray]] ]): system agents' actions.
            next_timestep (dm_env.TimeStep): data emitted by an environment during
                interaction.
            next_extras (Dict[str, types.NestedArray], optional): possible extra
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
        net_to_ints: Dict[str, int],
        executor_samples: List,
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
        self._executor_samples = executor_samples
        self._counts = counts
        self._net_to_ints = net_to_ints
        self._network_int_keys_extras: Dict[str, np.array] = {}
        self._cum_rewards: Dict[str, float] = {
            agent_key: 0.0 for agent_key in self._agent_specs.keys()
        }

        super().__init__(
            policy_networks=policy_networks,
            agent_net_keys=agent_net_keys,
            adder=adder,
            variable_client=variable_client,
            store_recurrent_state=store_recurrent_state,
        )

    # TODO (dries): This sometimes gives a warning. Is retracing a problem here?

    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
        state: types.NestedTensor,
        # is_batched: bool = False,
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
        # if not is_batched:
        batched_observation = tf2_utils.add_batch_dim(observation)
        # else:
        #     batched_observation = observation

        # index network either on agent type or on agent id
        agent_key = self._agent_net_keys[agent]

        # Compute the policy, conditioned on the observation.
        policy, new_state = self._policy_networks[agent_key](batched_observation, state)
        # TODO (dries): Make this support hybrid action spaces.
        if type(self._agent_specs[agent].actions) == BoundedArray:
            # Continuous action
            action = policy
        elif type(self._agent_specs[agent].actions) == DiscreteArray:
            action = tf.math.argmax(policy, axis=-1)
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

        # Step the recurrent policy forward given the current observation and state.
        action, policy, new_state = self._policy(
            agent, observation.observation, self._states[agent]
        )

        # Return a numpy array with squeezed out batch dimension.
        # TODO (dries): This is always a tensor. Maybe a tree
        # operation is not needed here?
        return action, policy, new_state

    # @staticmethod
    # def stack_all(s_list):
    #     return tree.map_structure(lambda *args:
    # tf.concat(list(args), axis=0), *s_list)

    # @staticmethod
    # def extract_state(nest, agent_i):
    #     return tree.map_structure(lambda nes:
    # tf2_utils.add_batch_dim(nes[agent_i]), nest)

    @tf.function
    def do_policies(self, observations: types.NestedArray) -> Tuple[Dict, Dict, Dict]:
        actions = {}
        policies = {}
        new_states = {}
        for agent, observation in observations.items():
            actions[agent], policies[agent], new_states[agent] = self.select_action(
                agent, observation
            )
        return actions, policies, new_states

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

        # Test for a special case of completely shared weights
        # if len(set(self._agent_net_keys.values()))==1:
        #     agents = sort_str_num(self._agent_net_keys.keys())
        #     @tf.function
        #     def do_this(observations, states, agents):
        #         batched_observations = tf.stack([observations[agent].
        # observation for agent in agents])
        #         batched_states = self.stack_all([states[agent] for agent in agents])

        #         tf_actions, tf_policies, new_states = self._policy(
        #             agents[0], batched_observations, batched_states,
        # is_batched=True,
        #         )

        #         return tf_actions, tf_policies, new_states

        #     tf_actions, tf_policies, new_states = do_this(observations,
        # self._states, agents)

        #     # Bookkeeping of recurrent states for the observe method.
        #     for a_i, agent in enumerate(agents):
        #         agent_state = self.extract_state(new_states, a_i)
        #         self._update_state(agent, agent_state)
        #         actions[agent] = tf_actions[a_i].numpy()
        #         policies[agent] = tf_policies[a_i].numpy()
        # else:
        # Use the usual sequential method to select actions.

        actions, policies, new_states = self.do_policies(observations)
        # actions = {}
        # policies = {}
        # for agent, observation in observations.items():
        #     actions[agent], policies[agent] = self.select_action(agent,
        # observation)
        # Bookkeeping of recurrent states for the observe method.
        self._states = new_states
        actions = tf2_utils.to_numpy_squeeze(actions)
        policies = tf2_utils.to_numpy_squeeze(policies)

        return actions, policies

    def _custom_end_of_episode_logic(self) -> None:
        """Custom logic at the end of an episode."""
        return

    def _add_to_cum_rewards(self, rewards: Dict[str, float]) -> None:
        for agent_key in rewards.keys():
            self._cum_rewards[agent_key] += rewards[agent_key]

    def sample_new_keys(self) -> None:
        """Sample new keys for the network ints."""
        agents = sort_str_num(list(self._agent_net_keys.keys()))
        self._network_int_keys_extras, self._agent_net_keys = sample_new_agent_keys(
            agents,
            self._executor_samples,
            self._net_to_ints,
        )

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

        # Re-initialize the RNN state.
        for agent, _ in timestep.observation.items():
            # index network either on agent type or on agent id
            agent_key = self._agent_net_keys[agent]
            self._states[agent] = self._policy_networks[agent_key].initial_state(1)

        # Sample new agent_net_keys.
        self.sample_new_keys()

        if not self._adder:
            return

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
            actions (Dict[str, types.NestedArray]): system agents' actions.
            next_timestep (dm_env.TimeStep): data emitted by an environment during
                interaction.
            next_extras (Dict[str, types.NestedArray]): possible extra
                information to record during the transition. Defaults to {}.
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
        self._add_to_cum_rewards(next_timestep.reward)
        self._adder.add(policy, next_timestep, next_extras)  # type: ignore
        # Custom end of episode logic.
        if next_timestep.last():
            self._custom_end_of_episode_logic()

    def update(self, wait: bool = False) -> None:
        """Update the policy variables."""
        if self._variable_client:
            self._variable_client.get_async()
