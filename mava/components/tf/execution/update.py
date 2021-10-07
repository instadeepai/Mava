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

<<<<<<< HEAD:mava/components/tf/execution/update.py
"""Commonly used adder components for system builders"""

from mava.callbacks import Callback
from mava.systems.execution import SystemExecutor


class Update(Callback):
    def on_execution_update(self, executor: SystemExecutor) -> None:
        """[summary]

        Args:
            executor (SystemExecutor): [description]
        """
        pass


class OnlineUpdate(Update):
    def on_execution_update(self, executor: SystemExecutor) -> None:
        if self._variable_client:
            self._variable_client.update(self._wait)
=======
from mava.systems.tf.mad4pg.execution import (
    MAD4PGFeedForwardExecutor,
    MAD4PGRecurrentExecutor,
)
from mava.systems.tf.mad4pg.networks import make_default_networks
from mava.systems.tf.mad4pg.system import MAD4PG
from mava.systems.tf.mad4pg.training import (
    MAD4PGBaseRecurrentTrainer,
    MAD4PGBaseTrainer,
    MAD4PGCentralisedRecurrentTrainer,
    MAD4PGCentralisedTrainer,
    MAD4PGDecentralisedRecurrentTrainer,
    MAD4PGDecentralisedTrainer,
    MAD4PGStateBasedRecurrentTrainer,
    MAD4PGStateBasedSingleActionCriticRecurrentTrainer,
    MAD4PGStateBasedTrainer,
)
>>>>>>> d3a272fe5f1343972503b4b0617c01d684baf5e4:mava/systems/tf/mad4pg/__init__.py
