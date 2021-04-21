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

"""Executor for MAPPO systems, using Tensorflow and Sonnet."""

from typing import Dict, Optional, Union

import dm_env
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
    """A feed-forward executor.
    An executor based on a feed-forward policy for each agent in the system
    which takes non-batched observations and outputs non-batched actions.
    It also allows adding experiences to replay and updating the weights
    from the policy on the learner.
    """

    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        shared_weights: bool = True,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ):
        """Initializes the executor.
        Args:
          policy_network: the policy to run for each agent in the system.
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
        self._prev_logits = None

    @tf.function
    def _policy(
        self, agent: str, observation: types.NestedTensor
    ) -> types.NestedTensor:

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(observation)

        # index network either on agent type or on agent id
        agent_key = agent.split("_")[0] if self._shared_weights else agent

        # Compute the policy, conditioned on the observation.
        logits = self._policy_networks[agent_key](batched_observation)
        action = tfd.Categorical(logits=logits).sample()

        self._prev_logits = logits

        return tf.cast(action, "int64")

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:

        # Pass the observation through the policy network.
        action = self._policy(agent, observation.observation)

        # TODO Mask actions here using observation.legal_actions
        # What happens in discrete vs cont case

        # Return a numpy array with squeezed out batch dimension.
        return tf2_utils.to_numpy_squeeze(action)

    def observe_first(
        self,
        timestep: dm_env.TimeStep,
        extras: Dict[str, types.NestedArray] = {"": ()},
    ) -> None:
        if self._adder:
            self._adder.add_first(timestep, extras)

    # TODO(Kale-ab) - This is temp. This will be changed when doing seq adder.
    def agent_observe_first(self, agent: str, timestep: dm_env.TimeStep) -> None:
        if self._adder:
            extras = {}
            extras["agent_id"] = agent
            self._adder.add_first(timestep, extras)

    # TODO(Kale-ab) - This is temp. This will be changed when doing seq adder.
    def agent_observe(
        self,
        agent: str,
        action: Union[float, int, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        extras: Optional[Dict] = {},
    ) -> None:
        if self._adder:
            if not extras:
                extras = {}
            extras["agent_id"] = agent
            extras["logits"] = self._prev_logits
            self._adder.add(action, next_timestep, extras)

    def observe(
        self,
        actions: Dict[str, types.NestedArray],
        next_timestep: dm_env.TimeStep,
        next_extras: Optional[Dict[str, types.NestedArray]] = {},
    ) -> None:
        if self._adder:
            if next_extras:
                self._adder.add(actions, next_timestep, next_extras)
            else:
                self._adder.add(actions, next_timestep)

    def select_actions(
        self, observations: Dict[str, types.NestedArray]
    ) -> Dict[str, types.NestedArray]:
        actions = {}
        for agent, observation in observations.items():
            # Pass the observation through the policy network.
            action = self._policy(agent, observation.observation)
            # TODO Mask actions here using observation.legal_actions
            # What happens in discrete vs cont case
            actions[agent] = tf2_utils.to_numpy_squeeze(action)

        # Return a numpy array with squeezed out batch dimension.

        return actions

    def update(self, wait: bool = False) -> None:
        if self._variable_client:
            self._variable_client.update(wait)
