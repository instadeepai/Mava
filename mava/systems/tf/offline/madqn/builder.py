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

"""Offline MADQN system builder implementation."""

import dataclasses
from mava.systems.tf.madqn.builder import MADQNConfig
from typing import Any, Dict, Iterator, List, Optional, Type, Union

import numpy as np
import reverb
import sonnet as snt
from acme import datasets
from acme.tf import variable_utils
from acme.utils import counting

from mava.components.tf.recorder import TFRecordLoader
from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.components.tf.modules.stabilising import FingerPrintStabalisation
from mava.systems.tf import executors
from mava.systems.tf.offline.madqn import execution, training 
from mava.wrappers import MADQNDetailedTrainerStatistics
from mava.systems.tf.madqn.builder import MADQNConfig, MADQNBuilder


@dataclasses.dataclass
class OfflineMADQNConfig(MADQNConfig):
    """Configuration options for the MADQN system.
    Args:
        environment_spec: description of the action and observation spaces etc. for
            each agent in the system.
        epsilon_min: final minimum value for epsilon at the end of a decay schedule.
        epsilon_decay: the rate at which epislon decays.
        agent_net_keys: (dict, optional): specifies what network each agent uses.
            Defaults to {}.
        target_update_period: number of learner steps to perform before updating
            the target networks.
        executor_variable_update_period: the rate at which executors sync their
            paramters with the trainer.
        max_gradient_norm: value to specify the maximum clipping value for the gradient
            norm during optimization.
        min_replay_size: minimum replay size before updating.
        max_replay_size: maximum replay size.
        samples_per_insert: number of samples to take from replay for every insert
            that is made.
        prefetch_size: size to prefetch from replay.
        batch_size: batch size for updates.
        n_step: number of steps to include prior to boostrapping.
        sequence_length: recurrent sequence rollout length.
        period: consecutive starting points for overlapping rollouts across a sequence.
        discount: discount to use for TD updates.
        checkpoint: boolean to indicate whether to checkpoint models.
        optimizer: type of optimizer to use for updating the parameters of models.
        replay_table_name: string indicating what name to give the replay table.
        checkpoint_subpath: subdirectory specifying where to store checkpoints."""
    reverb_checkpoint_dir: str 


class OfflineMADQNBuilder(MADQNBuilder):
    """Builder for MADQN which constructs individual components of the system."""

    def __init__(
        self,
        config: OfflineMADQNConfig,
        trainer_fn: Type[training.OfflineMADQNTrainer] = training.OfflineMADQNTrainer,
        executor_fn: Type[core.Executor] = execution.OfflineMADQNFeedForwardExecutor,
        extra_specs: Dict[str, Any] = {},
        exploration_scheduler_fn: Type[
            LinearExplorationScheduler
        ] = LinearExplorationScheduler,
        replay_stabilisation_fn: Optional[Type[FingerPrintStabalisation]] = None,
    ):
        """Initialise the system.

        Args:
            config (MADQNConfig): system configuration specifying hyperparameters and
                additional information for constructing the system.
            trainer_fn (Type[training.MADQNTrainer], optional): Trainer function, of a
                correpsonding type to work with the selected system architecture.
                Defaults to training.MADQNTrainer.
            executor_fn (Type[core.Executor], optional): Executor function, of a
                corresponding type to work with the selected system architecture.
                Defaults to execution.MADQNFeedForwardExecutor.
            extra_specs (Dict[str, Any], optional): defines the specifications of extra
                information used by the system. Defaults to {}.
            exploration_scheduler_fn (Type[ LinearExplorationScheduler ], optional):
                epsilon decay scheduler. Defaults to LinearExplorationScheduler.
            replay_stabilisation_fn (Optional[Type[FingerPrintStabalisation]],
                optional): optional function to stabilise experience replay. Defaults
                to None.
        """
        super().__init__(
            config,
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
            exploration_scheduler_fn=exploration_scheduler_fn,
            replay_stabilisation_fn=replay_stabilisation_fn,
        )

    def make_reverb_checkpointer(self):
        return reverb.checkpointers.DefaultCheckpointer(path=self._config.reverb_checkpoint_dir)

    def make_replay_tables(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into.

        Args:
            environment_spec (specs.MAEnvironmentSpec): description of the action and
                observation spaces etc. for each agent in the system.

        Raises:
            NotImplementedError: unknown executor type.

        Returns:
            List[reverb.Table]: a list of data tables for inserting data.
        """

        adder_sig = self._make_adder_signiture(environment_spec)

        # When training offline we only use a MinSize RateLimiter
        # because no new samples will be added to reverb.
        rate_limiter = reverb.rate_limiters.MinSize(1)

        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=rate_limiter,
            signature=adder_sig,
        )

        return [replay_table]

    def make_dataset_iterator(self, replay_client: reverb.Client) -> Iterator[reverb.ReplaySample]:
        loader = TFRecordLoader(self._agents)
        dataset = loader.as_tf_dataset()
        return iter(dataset)
