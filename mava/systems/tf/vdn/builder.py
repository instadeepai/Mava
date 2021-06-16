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
from mava.wrappers import DetailedTrainerStatisticsWithEpsilon

# TODO Clean up documentation


@dataclasses.dataclass
class VDNConfig(MADQNConfig):
    # TODO Match documentation with MADQNConfig once cleaned.
    """Configuration options for the QMIX system.
    Args:
            environment_spec: description of the actions, observations, etc.
            discount: discount to use for TD updates.
            batch_size: batch size for updates.
            prefetch_size: size to prefetch from replay.
            target_update_period: number of learner steps to perform before updating
              the target networks.
            min_replay_size: minimum replay size before updating.
            max_replay_size: maximum replay size.
            samples_per_insert: number of samples to take from replay for every insert
              that is made.
            n_step: number of steps to squash into a single transition.
            sigma: standard deviation of zero-mean, Gaussian exploration noise.
            replay_table_name: string indicating what name to give the replay table."""


class VDNBuilder(MADQNBuilder):
    """Builder for VDN which constructs individual components of the system."""

    """Defines an interface for defining the components of an MARL system.
      Implementations of this interface contain a complete specification of a
      concrete MARL system. An instance of this class can be used to build an
      MARL system which interacts with the environment either locally or in a
      distributed setup.
      """

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
        """Args:
        _config: Configuration options for the QMIX system.
        _trainer_fn: Trainer module to use.
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
    ) -> core.Trainer:
        """Creates an instance of the trainer.
        Args:
          networks: struct describing the networks needed by the trainer; this can
            be specific to the trainer in question.
          dataset: iterator over samples from replay.
          counter: a Counter which allows for recording of counts (trainer steps,
            executor steps, etc.) distributed throughout the system.
          logger: Logger object for logging metadata.
        """
        q_networks = networks["values"]
        target_q_networks = networks["target_values"]
        mixing_network = networks["mixing"]
        target_mixing_network = networks["target_mixing"]

        agents = self._config.environment_spec.get_agent_ids()
        agent_types = self._config.environment_spec.get_agent_types()

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
            shared_weights=self._config.shared_weights,
            optimizer=self._config.optimizer,
            target_update_period=self._config.target_update_period,
            max_gradient_norm=self._config.max_gradient_norm,
            exploration_scheduler=exploration_scheduler,
            communication_module=communication_module,
            dataset=dataset,
            counter=counter,
            fingerprint=fingerprint,
            logger=logger,
            checkpoint=self._config.checkpoint,
            checkpoint_subpath=self._config.checkpoint_subpath,
        )

        trainer = DetailedTrainerStatisticsWithEpsilon(trainer)  # type:ignore

        return trainer
