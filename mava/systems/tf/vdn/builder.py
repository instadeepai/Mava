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

"""VDN system builder implementation."""

import dataclasses
from typing import Any, Dict, Iterator, Optional, Type

import reverb
import sonnet as snt
from acme.utils import counting

from mava import core, types
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.components.tf.modules.exploration.exploration_scheduling import (
    LinearExplorationScheduler,
)
from mava.components.tf.modules.stabilising import FingerPrintStabalisation
from mava.systems.tf.madqn.builder import MADQNBuilder, MADQNConfig
from mava.systems.tf.vdn import execution, training
from mava.wrappers import MADQNDetailedTrainerStatistics


@dataclasses.dataclass
class VDNConfig(MADQNConfig):
    """Configuration options for the VDN system.
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
        checkpoint_minute_interval (int): The number of minutes to wait between
                checkpoints.
        optimizer: type of optimizer to use for updating the parameters of models.
        replay_table_name: string indicating what name to give the replay table.
        checkpoint_subpath: subdirectory specifying where to store checkpoints."""


class VDNBuilder(MADQNBuilder):
    """Builder for VDN which constructs individual components of the system."""

    def __init__(
        self,
        config: VDNConfig,
        trainer_fn: Type[training.VDNTrainer] = training.VDNTrainer,
        executor_fn: Type[core.Executor] = execution.VDNFeedForwardExecutor,
        extra_specs: Dict[str, Any] = {},
        exploration_scheduler_fn: Type[
            LinearExplorationScheduler
        ] = LinearExplorationScheduler,
        replay_stabilisation_fn: Optional[Type[FingerPrintStabalisation]] = None,
    ) -> None:
        """Initialise the system.

        Args:
            config (VDNConfig): system configuration specifying hyperparameters and
                additional information for constructing the system.
            trainer_fn (Type[training.VDNTrainer], optional): Trainer function, of a
                correpsonding type to work with the selected system architecture.
                Defaults to training.VDNTrainer.
            executor_fn (Type[core.Executor], optional): Executor function, of a
                corresponding type to work with the selected system architecture.
                Defaults to execution.VDNFeedForwardExecutor.
            extra_specs (Dict[str, Any], optional): defines the specifications of extra
                information used by the system. Defaults to {}.
            exploration_scheduler_fn (Type[ LinearExplorationScheduler ], optional):
                epsilon decay scheduler. Defaults to LinearExplorationScheduler.
            replay_stabilisation_fn (Optional[Type[FingerPrintStabalisation]],
                optional): optional function to stabilise experience replay. Defaults
                to None.
        """
        super(VDNBuilder, self).__init__(
            config=config,
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
            exploration_scheduler_fn=exploration_scheduler_fn,
            replay_stabilisation_fn=replay_stabilisation_fn,
        )

    def make_trainer(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: Iterator[reverb.ReplaySample],
        counter: Optional[counting.Counter] = None,
        logger: Optional[types.NestedLogger] = None,
        communication_module: Optional[BaseCommunicationModule] = None,
        replay_client: Optional[reverb.TFClient] = None,
    ) -> core.Trainer:
        """Create a trainer instance.

        Args:
            networks (Dict[str, Dict[str, snt.Module]]): system networks.
            dataset (Iterator[reverb.ReplaySample]): dataset iterator to feed data to
                the trainer networks.
            counter (Optional[counting.Counter], optional): a Counter which allows for
                recording of counts, e.g. trainer steps. Defaults to None.
            logger (Optional[types.NestedLogger], optional): Logger object for logging
                metadata.. Defaults to None.
            communication_module (BaseCommunicationModule): module to enable
                agent communication. Defaults to None.
            replay_client (reverb.TFClient): Used for importance sampling.
                Not implemented yet.

        Returns:
            core.Trainer: system trainer, that uses the collected data from the
                executors to update the parameters of the agent networks in the system.
        """

        agents = self._config.environment_spec.get_agent_ids()
        agent_types = self._config.environment_spec.get_agent_types()

        q_networks = networks["values"]
        target_q_networks = networks["target_values"]

        mixing_network = networks["mixing"]
        target_mixing_network = networks["target_mixing"]

        # Make epsilon scheduler
        exploration_scheduler = self._exploration_scheduler_fn(
            epsilon_min=self._config.epsilon_min,
            epsilon_decay=self._config.epsilon_decay,
        )

        # Check if we should use fingerprints
        fingerprint = True if self._replay_stabiliser_fn is not None else False

        # The learner updates the parameters (and initializes them).
        trainer = self._trainer_fn(  # type:ignore
            agents=agents,
            agent_types=agent_types,
            discount=self._config.discount,
            q_networks=q_networks,
            target_q_networks=target_q_networks,
            mixing_network=mixing_network,
            target_mixing_network=target_mixing_network,
            agent_net_keys=self._config.agent_net_keys,
            optimizer=self._config.optimizer,
            target_update_period=self._config.target_update_period,
            max_gradient_norm=self._config.max_gradient_norm,
            exploration_scheduler=exploration_scheduler,
            communication_module=communication_module,
            dataset=dataset,
            counter=counter,
            fingerprint=fingerprint,
            logger=logger,
            checkpoint_minute_interval=self._config.checkpoint_minute_interval,
            checkpoint=self._config.checkpoint,
            checkpoint_subpath=self._config.checkpoint_subpath,
        )

        trainer = MADQNDetailedTrainerStatistics(trainer)  # type:ignore

        return trainer
