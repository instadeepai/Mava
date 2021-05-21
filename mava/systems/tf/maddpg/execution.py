# python3
# Copyright 2021 [...placeholder...]. All rights reserved.
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
from mava.systems import executors

tfd = tfp.distributions


class MADDPGDiscreteFeedForwardExecutor(executors.FeedForwardExecutor):
    """A feed-forward executor for discrete actions in MADDPG.
    An executor based on a feed-forward policy for each agent in the system
    which takes non-batched observations and outputs non-batched actions.
    It also allows adding experiences to replay and updating the weights
    from the policy on the learner.
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
        self.self._policy_outputs: Dict[str, Any] = {}

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

        # Sample from the policy if it is stochastic.
        action = tf.math.argmax(policy)

        return action, policy

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:

        # Step the recurrent policy/value network forward
        # given the current observation and state.
        action, self._policy_outputs[agent] = self._policy(agent, observation)

        # Return a numpy array with squeezed out batch dimension.
        action = tf2_utils.to_numpy_squeeze(action)
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

            # Generate dummy policy values to send through
            self.select_actions(timestep.observation)
            extras.update({"policy_out": self._policy_outputs})
            self._adder.add_first(timestep)

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Dict[str, types.NestedArray] = {},
    ) -> None:

        if not self._adder:
            return

        next_extras.update({"policy_out": self._policy_outputs})

        next_extras = tf2_utils.to_numpy_squeeze(next_extras)

        self._adder.add(actions, next_timestep, next_extras)

    def update(self, wait: bool = False) -> None:
        if self._variable_client:
            self._variable_client.update(wait)
