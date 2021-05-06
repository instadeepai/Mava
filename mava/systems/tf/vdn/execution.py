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

# TODO (StJohn): implement VDN executor (if required)
# Helper resources
#   - single agent generic actors in acme:
#           https://github.com/deepmind/acme/blob/master/acme/agents/tf/actors.py
#   - single agent custom actor for Impala in acme:
#           https://github.com/deepmind/acme/blob/master/acme/agents/tf/impala/acting.py
#   - multi-agent generic executors in mava: mava/systems/tf/executors.py
#

from typing import Dict, Optional

import sonnet as snt
import tensorflow as tf
from acme import types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.systems.tf import executors


class VDNFeedForwardExecutor(executors.FeedForwardExecutor):
    def __init__(
        self,
        policy_networks: Dict[str, snt.Module],
        shared_weights: bool = True,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
    ) -> None:
        super().__init__(policy_networks, shared_weights, adder, variable_client)

    @tf.function
    def _policy(
        self, agent: str, observation: types.NestedTensor
    ) -> types.NestedTensor:
        # Add a dummy batch dimension and as a side effect convert numpy to TF.
        batched_observation = tf2_utils.add_batch_dim(
            tf.dtypes.cast(observation, tf.float32)
        )

        # index network either on agent type or on agent id
        agent_key = agent.split("_")[0] if self._shared_weights else agent

        # Compute the policy, conditioned on the observation.
        policy = self._policy_networks[agent_key](batched_observation)

        # Sample from the policy if it is stochastic.
        # action = policy.sample() if isinstance(policy, tfd.Distribution) else policy
        action = tf.argmax(policy, axis=1)

        return action
