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
import sonnet as snt
import tensorflow as tf
import tensorflow_probability as tfp
from acme import types

# Internal imports.
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders, core

tfd = tfp.distributions


class MAPPORecurrentExecutor(core.Executor):
    """A recurrent Executor for MAPPO.
    An executor based on a recurrent policy for each agent in the system which
    takes non-batched observations and outputs non-batched actions, and keeps
    track of the recurrent state inside. It also allows adding experiences to
    replay and updating the weights from the policy on the learner.
    """

    def __init__(
        self,
        networks: Dict[str, snt.RNNCore],
        shared_weights: bool = False,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
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
        self._networks = networks
        self._shared_weights = shared_weights

        self._states: Dict[str, Any] = {net_key: None for net_key in networks.keys()}
        self._prev_states: Dict[str, Any] = {
            net_key: None for net_key in networks.keys()
        }
        self._prev_logits: Dict[str, Any] = {
            net_key: None for net_key in networks.keys()
        }

    def _policy(
        self,
        agent_key: str,
        observation: types.NestedTensor,
        state: types.NestedTensor,
    ) -> Tuple[types.NestedTensor, types.NestedTensor]:

        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        observation = tf2_utils.add_batch_dim(observation)

        # Compute the policy, conditioned on the observation.
        logits, new_state = self._networks[agent_key](observation, state)

        return logits, new_state

    def select_action(
        self, agent: str, observation: types.NestedArray
    ) -> types.NestedArray:

        # Index network either on agent type or on agent id.
        agent_key = agent.split("_")[0] if self._shared_weights else agent

        # Initialize the RNN state if necessary.
        if self._states[agent] is None:
            self._states[agent] = self._networks[agent_key].initial_state(1)

        # Step the recurrent policy/value network forward
        # given the current observation and state.
        (logits, _), new_state = self._policy(
            agent_key, observation.observation, self._states[agent]
        )

        # Bookkeeping of recurrent states for the observe method.
        self._prev_logits[agent_key] = logits
        self._prev_states[agent_key] = self._states[agent_key]
        self._states[agent_key] = new_state

        # Sample action
        action = tfd.Categorical(logits).sample()
        action = tf.cast(action, dtype="int64")
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

        # Set the state to None so that we re-initialize at the next policy call.
        for agent, _ in timestep.observation.items():
            # Index either on agent type or on agent id
            agent_key = agent.split("_")[0] if self._shared_weights else agent
            self._states[agent_key] = None

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

        next_extras.update(
            {"logits": self._prev_logits, "core_states": self._prev_states}
        )

        next_extras = tf2_utils.to_numpy_squeeze(next_extras)

        self._adder.add(actions, next_timestep, next_extras)

    def update(self, wait: bool = False) -> None:
        if self._variable_client:
            self._variable_client.update(wait)
