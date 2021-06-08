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

from typing import Dict, List, Optional, Tuple, Union

import dm_env
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

Array = specs.Array
BoundedArray = specs.BoundedArray
DiscreteArray = specs.DiscreteArray

tfd = tfp.distributions


class MADDPGFeedForwardExecutor(executors.FeedForwardExecutor):
    """A feed-forward executor for discrete actions in MADDPG.
    An executor based on a feed-forward policy for each agent in the system
    which takes non-batched observations and outputs non-batched actions.
    It also allows adding experiences to replay and updating the weights
    from the policy on the learner.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        shared_weights: bool = True,
    ):

        """Initializes the executor.
        Args:
          networks: the (recurrent) policy to run for each agent in the system.
          shared_weights: specify if weights are shared between agent networks.
          adder: the adder object to which allows to add experiences to a
            dataset/replay buffer.
          variable_client: object which allows to copy weights from the trainer copy
            of the policies to the executor copy (in case they are separate).
        """

        # Store these for later use.
        self._agent_specs = agent_specs
        super().__init__(
            policy_networks=policy_networks,
            shared_weights=shared_weights,
            adder=adder,
            variable_client=variable_client,
        )

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

        actions = {}
        policies = {}
        for agent, observation in observations.items():
            actions[agent], policies[agent] = self.select_action(agent, observation)
        return actions, policies

    def observe(
        self,
        actions: Union[
            Dict[str, types.NestedArray], List[Dict[str, types.NestedArray]]
        ],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        if self._adder:
            _, policy = actions
            # TODO (dries): Sort out this mypy issue.
            self._adder.add(policy, next_timestep, next_extras)  # type: ignore


class MADDPGRecurrentExecutor(executors.RecurrentExecutor):
    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_specs: Dict[str, EnvironmentSpec],
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        shared_weights: bool = True,
        store_recurrent_state: bool = True,
    ):

        """Initializes the executor.
        Args:
          networks: the (recurrent) policy to run for each agent in the system.
          shared_weights: specify if weights are shared between agent networks.
          adder: the adder object to which allows to add experiences to a
            dataset/replay buffer.
          variable_client: object which allows to copy weights from the trainer copy
            of the policies to the executor copy (in case they are separate).
        """

        # Store these for later use.
        self._agent_specs = agent_specs

        super().__init__(
            policy_networks=policy_networks,
            shared_weights=shared_weights,
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

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # index network either on agent type or on agent id
        agent_key = agent.split("_")[0] if self._shared_weights else agent

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

        # TODO Mask actions here using observation.legal_actions
        # Initialize the RNN state if necessary.
        if self._states[agent] is None:
            # index network either on agent type or on agent id
            agent_key = agent.split("_")[0] if self._shared_weights else agent
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
