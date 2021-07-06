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

"""QMIX system executor implementation."""

from typing import Dict, Optional

import sonnet as snt
from acme.tf import variable_utils as tf2_variable_utils

from mava import adders
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.systems.tf.madqn.execution import MADQNFeedForwardExecutor
from mava.systems.tf.madqn.training import MADQNTrainer


class QMIXFeedForwardExecutor(MADQNFeedForwardExecutor):
    """A feed-forward executor.
    An executor based on a feed-forward policy for each agent in the system.
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
        """Initialise the system executor

        Args:
            q_networks (Dict[str, snt.Module]): q-value networks for each agent in the
                system.
            action_selectors (Dict[str, Any]): policy action selector method, e.g.
                epsilon greedy.
            trainer (MADQNTrainer, optional): system trainer.
            shared_weights (bool, optional): whether agents should share weights or not.
                Defaults to True.
            adder (Optional[adders.ParallelAdder], optional): adder which sends data
                to a replay buffer. Defaults to None.
            variable_client (Optional[tf2_variable_utils.VariableClient], optional):
                client to copy weights from the trainer. Defaults to None.
            communication_module (BaseCommunicationModule): module for enabling
                communication protocols between agents. Defaults to None.
            fingerprint (bool, optional): whether to use fingerprint stabilisation to
                stabilise experience replay. Defaults to False.
            evaluator (bool, optional): whether the executor will be used for
                evaluation. Defaults to False.
        """

        super(QMIXFeedForwardExecutor, self).__init__(
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
