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

"""DIAL system builder implementation."""

import dataclasses
from typing import Any, Dict, Iterator, Optional, Type, Union

import reverb
import sonnet as snt
from acme.utils import counting

from mava import adders, core, types
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.components.tf.modules.exploration.exploration_scheduling import (
    BaseExplorationScheduler,
    BaseExplorationTimestepScheduler,
    ConstantScheduler,
)
from mava.components.tf.modules.stabilising import FingerPrintStabalisation
from mava.systems.tf.madqn import execution, training
from mava.systems.tf.madqn.builder import MADQNBuilder, MADQNConfig


@dataclasses.dataclass
class DIALConfig(MADQNConfig):
    """Configuration options for the DIAL system.

    Args:
        environment_spec: description of the action and observation spaces etc. for
            each agent in the system.
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
        checkpoint_subpath: subdirectory specifying where to store checkpoints.
        learning_rate_scheduler_fn: function/class that takes in a trainer step t
                and returns the current learning rate.
    """


class DIALBuilder(MADQNBuilder):
    """Builder for DIAL which constructs individual components of the system."""

    def __init__(
        self,
        config: DIALConfig,
        trainer_fn: Type[
            training.MADQNRecurrentCommTrainer
        ] = training.MADQNRecurrentCommTrainer,
        executor_fn: Type[core.Executor] = execution.MADQNRecurrentCommExecutor,
        extra_specs: Dict[str, Any] = {},
        replay_stabilisation_fn: Optional[Type[FingerPrintStabalisation]] = None,
    ):
        """Initialise the system.

        Args:
            config (DIALConfig): system configuration specifying hyperparameters and
                additional information for constructing the system.
            trainer_fn (Type[ training.MADQNRecurrentCommTrainer ], optional):
                Trainer function, of a correpsonding type to work with the selected
                system architecture. Defaults to training.MADQNRecurrentCommTrainer.
            executor_fn (Type[core.Executor], optional): Executor function, of a
                corresponding type to work with the selected system architecture.
                Defaults to execution.MADQNRecurrentCommExecutor.
            extra_specs (Dict[str, Any], optional): defines the specifications of extra
                information used by the system. Defaults to {}.
            replay_stabilisation_fn (Optional[Type[FingerPrintStabalisation]],
                optional): optional function to stabilise experience replay. Defaults
                to None.
        """
        super(DIALBuilder, self).__init__(
            config=config,
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
            replay_stabilisation_fn=replay_stabilisation_fn,
        )

    def make_executor(  # type: ignore[override]
        self,
        q_networks: Dict[str, snt.Module],
        exploration_schedules: Dict[
            str,
            Union[
                BaseExplorationTimestepScheduler,
                BaseExplorationScheduler,
                ConstantScheduler,
            ],
        ],
        action_selectors: Dict[str, Any],
        communication_module: BaseCommunicationModule,
        adder: Optional[adders.ReverbParallelAdder] = None,
        variable_source: Optional[core.VariableSource] = None,
        trainer: Optional[training.MADQNRecurrentCommTrainer] = None,
        evaluator: bool = False,
        seed: Optional[int] = None,
    ) -> core.Executor:
        """Create an executor instance.

        Args:
            q_networks (Dict[str, snt.Module]): q-value networks for each agent in the
                system.
            exploration_schedules : epsilon decay schedule per agent.
            action_selectors (Dict[str, Any]): policy action selector method, e.g.
                epsilon greedy.
            communication_module (BaseCommunicationModule): module for enabling
                communication protocols between agents.
            adder (Optional[adders.ReverbParallelAdder], optional): adder to send data
                to a replay buffer. Defaults to None.
            variable_source (Optional[core.VariableSource], optional): variables server.
                Defaults to None.
            trainer (Optional[training.MADQNRecurrentCommTrainer], optional):
                system trainer. Defaults to None.
            evaluator (bool, optional): boolean indicator if the executor is used for
                for evaluation only. Defaults to False.
            seed: seed for reproducible sampling.

        Returns:
            core.Executor: system executor, a collection of agents making up the part
                of the system generating data by interacting the environment.
        """

        return super().make_executor(
            q_networks=q_networks,
            exploration_schedules=exploration_schedules,
            action_selectors=action_selectors,
            communication_module=communication_module,
            adder=adder,
            variable_source=variable_source,
            trainer=trainer,
            evaluator=evaluator,
            seed=seed,
        )

    def make_trainer(  # type: ignore[override]
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: Iterator[reverb.ReplaySample],
        communication_module: BaseCommunicationModule,  # type: ignore
        counter: Optional[counting.Counter] = None,
        logger: Optional[types.NestedLogger] = None,
        replay_client: Optional[reverb.TFClient] = None,
    ) -> core.Trainer:
        """Create a trainer instance.

        Args:
            networks (Dict[str, Dict[str, snt.Module]]): system networks.
            dataset (Iterator[reverb.ReplaySample]): dataset iterator to feed data to
                the trainer networks.
            communication_module (BaseCommunicationModule): module to enable
                agent communication.
            counter (Optional[counting.Counter], optional): a Counter which allows for
                recording of counts, e.g. trainer steps. Defaults to None.
            logger (Optional[types.NestedLogger], optional): Logger object for logging
                metadata.. Defaults to None.
            replay_client (reverb.TFClient): Used for importance sampling.
                Not implemented yet.

        Returns:
            core.Trainer: system trainer, that uses the collected data from the
                executors to update the parameters of the agent networks in the system.
        """
        return super().make_trainer(
            networks=networks,
            dataset=dataset,
            communication_module=communication_module,
            counter=counter,
            logger=logger,
            replay_client=replay_client,
        )
