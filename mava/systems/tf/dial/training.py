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


"""DIAL trainer implementation."""

from typing import Dict, List

import sonnet as snt
import tensorflow as tf
from acme.utils import counting, loggers

from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.systems.tf.madqn.training import RecurrentCommMADQNTrainer


class DIALTrainer(RecurrentCommMADQNTrainer):
    """Recurrent Comm DIAL trainer.
    This is the trainer component of a MADQN system. IE it takes a dataset as input
    and implements update functionality to learn from this dataset.
    """

    def __init__(
        self,
        agents: List[str],
        agent_types: List[str],
        q_networks: Dict[str, snt.Module],
        target_q_networks: Dict[str, snt.Module],
        target_update_period: int,
        dataset: tf.data.Dataset,
        optimizer: snt.Optimizer,
        discount: float,
        shared_weights: bool,
        exploration_scheduler: LinearExplorationScheduler,
        communication_module: BaseCommunicationModule,
        clipping: bool = True,
        fingerprint: bool = False,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
    ):
        super().__init__(
            agents=agents,
            agent_types=agent_types,
            q_networks=q_networks,
            target_q_networks=target_q_networks,
            target_update_period=target_update_period,
            dataset=dataset,
            optimizer=optimizer,
            discount=discount,
            shared_weights=shared_weights,
            exploration_scheduler=exploration_scheduler,
            clipping=clipping,
            fingerprint=fingerprint,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
            communication_module=communication_module,
        )