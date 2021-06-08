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

"""Executor implementations for VDN."""

from typing import Dict, Optional

import sonnet as snt
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.systems.tf.madqn.execution import MADQNFeedForwardExecutor
from mava.systems.tf.madqn.training import MADQNTrainer


class VDNFeedForwardExecutor(MADQNFeedForwardExecutor):
    """A feed-forward executor.
    An executor based on a feed-forward policy for each agent in the system
    which takes non-batched observations and outputs non-batched actions.
    It also allows adding experiences to replay and updating the weights
    from the policy on the learner.
    """

    def __init__(
        self,
        q_networks: Dict[str, snt.Module],
        action_selectors: Dict[str, snt.Module],
        trainer: MADQNTrainer,
        shared_weights: bool = True,
        adder: Optional[adders.ParallelAdder] = None,
        variable_client: Optional[tf2_variable_utils.VariableClient] = None,
        communication_module: Optional[BaseCommunicationModule] = None,
        fingerprint: bool = False,
        evaluator: bool = False,
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
        super(VDNFeedForwardExecutor, self).__init__(
            q_networks=q_networks,
            action_selectors=action_selectors,
            shared_weights=shared_weights,
            adder=adder,
            variable_client=variable_client,
            communication_module=communication_module,
            fingerprint=fingerprint,
            trainer=trainer,
            evaluator=evaluator,
        )
