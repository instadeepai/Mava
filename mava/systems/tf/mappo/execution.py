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

from typing import Any, Dict, Optional, Tuple

import dm_env
import numpy as np
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme import types

# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders, core

tfd = tfp.distributions


class MAPPOFeedForwardExecutor(core.Executor):
    """A recurrent Executor for MAPPO.
    An executor based on a recurrent policy for each agent in the system which
    takes non-batched observations and outputs non-batched actions, and keeps
    track of the recurrent state inside. It also allows adding experiences to
    replay and updating the weights from the policy on the learner.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
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
        self._adder = adder
        self._variable_client = variable_client
        self._policy_networks = policy_networks
        self._shared_weights = shared_weights
        self._prev_log_probs: Dict[str, Any] = {}

    @tf.function
    def _policy(
        self,
        agent: str,
        observation: types.NestedTensor,
    ) -> Tuple[types.NestedTensor, types.NestedTensor]:
        # Index network either on agent type or on agent id.
        network_key = agent.split("_")[0] if self._shared_weights else agent

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

        if self._adder:
            self._adder.add_first(timestep)

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:

        if not self._adder:
            return

        next_extras.update({"log_probs": self._prev_log_probs})

        next_extras = tf2_utils.to_numpy_squeeze(next_extras)

        self._adder.add(actions, next_timestep, next_extras)

    def update(self, wait: bool = False) -> None:
        if self._variable_client:
            self._variable_client.update(wait)
