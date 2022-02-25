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

from typing import Any, Dict, Optional, Tuple

import dm_env
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
import tree
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.systems.tf import executors
from mava.types import OLT
from mava.utils.training_utils import action_mask_categorical_policies

tfd = tfp.distributions


class MAPPOFeedForwardExecutor(executors.FeedForwardExecutor):
    """A feed-forward executor.

    An executor based on a feed-forward policy for each agent in the system.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        agent_net_keys: Dict[str, str],
        adder: Optional[adders.ReverbParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        evaluator: bool = False,
        interval: Optional[dict] = None,
    ):
        """Initialise the system executor

        Args:
            policy_networks: policy networks for each agent in
                the system.
            agent_net_keys: specifies what network each agent uses.
            adder: adder which sends data to a replay buffer.
            variable_client: client to copy weights from the trainer.
            evaluator: whether the executor will be used for
                evaluation. Defaults to False.
            interval: interval that evaluations are run at.
        """

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

        Returns:
            Tuple[types.NestedTensor, types.NestedTensor]: log probabilities and action
        """

        # Index network either on agent type or on agent id.
        network_key = self._agent_net_keys[agent]

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        observation = tf2_utils.add_batch_dim(observation_olt.observation)

        # Compute the policy, conditioned on the observation.
        policy = self._policy_networks[network_key](observation)

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
        return log_prob, action

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
            log_prob, action = self._policy(agent, observation)

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
        self._prev_log_probs = tree.map_structure(tf2_utils.to_numpy_squeeze, log_probs)
        return actions

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {},
    ) -> None:
        """Record first observed timestep from the environment

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

        next_extras.update({"log_probs": self._prev_log_probs})

        self._adder.add(actions, next_timestep, next_extras)

    def update(self, wait: bool = False) -> None:
        """Update executor variables

        Args:
            wait (bool, optional): whether to stall the executor's request for new
                variables. Defaults to False.
        """

        if self._variable_client:
            self._variable_client.update(wait)
