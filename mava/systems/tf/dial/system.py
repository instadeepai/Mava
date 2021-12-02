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

"""DIAL system implementation."""

from typing import Any, Callable, Dict, Optional, Type, Union

import acme
import dm_env
import reverb
import sonnet as snt
from acme import specs as acme_specs
from acme.utils import counting

import mava
from mava import core
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedValueActor
from mava.components.tf.modules.communication import (
    BaseCommunicationModule,
    BroadcastedCommunication,
)
from mava.components.tf.modules.stabilising import FingerPrintStabalisation
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf.dial import builder
from mava.systems.tf.dial.execution import DIALSwitchExecutor
from mava.systems.tf.dial.training import DIALSwitchTrainer
from mava.systems.tf.madqn import training
from mava.systems.tf.madqn.system import MADQN
from mava.types import EpsilonScheduler
from mava.utils.loggers import MavaLogger


class DIAL(MADQN):
    """DIAL system."""

    def __init__(
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        exploration_scheduler_fn: Union[
            EpsilonScheduler,
            Dict[str, EpsilonScheduler],
            Dict[str, Dict[str, EpsilonScheduler]],
        ],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[DecentralisedValueActor] = DecentralisedValueActor,
        trainer_fn: Type[training.MADQNRecurrentCommTrainer] = DIALSwitchTrainer,
        communication_module: Type[BaseCommunicationModule] = BroadcastedCommunication,
        executor_fn: Type[core.Executor] = DIALSwitchExecutor,
        replay_stabilisation_fn: Optional[Type[FingerPrintStabalisation]] = None,
        num_executors: int = 1,
        num_caches: int = 0,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        agent_net_keys: Dict[str, str] = {},
        batch_size: int = 256,
        prefetch_size: int = 4,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: Optional[float] = 4.0,
        n_step: int = 5,
        sequence_length: int = 20,
        period: int = 20,
        importance_sampling_exponent: Optional[float] = None,
        max_priority_weight: float = 0.9,
        max_gradient_norm: float = None,
        discount: float = 1,
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]] = snt.optimizers.Adam(
            learning_rate=1e-4
        ),
        target_update_period: int = 100,
        executor_variable_update_period: int = 1000,
        max_executor_steps: int = None,
        checkpoint: bool = False,
        checkpoint_subpath: str = "~/mava/",
        checkpoint_minute_interval: int = 5,
        logger_config: Dict = {},
        train_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        train_loop_fn_kwargs: Dict = {},
        eval_loop_fn_kwargs: Dict = {},
        evaluator_interval: Optional[dict] = None,
        learning_rate_scheduler_fn: Optional[Callable[[int], None]] = None,
        seed: Optional[int] = None,
    ):
        """Initialise the system

        Args:
            environment_factory (Callable[[bool], dm_env.Environment]): function to
                instantiate an environment.
            network_factory (Callable[[acme_specs.BoundedArray],
                Dict[str, snt.Module]]): function to instantiate system networks.
            logger_factory (Callable[[str], MavaLogger], optional): function to
                instantiate a system logger. Defaults to None.
            architecture (Type[DecentralisedValueActor], optional): system architecture,
                e.g. decentralised or centralised. Defaults to DecentralisedValueActor.
            trainer_fn (Type[ training.MADQNRecurrentCommTrainer ], optional):
                training type associated with executor and architecture, e.g.
                centralised training. Defaults to training.MADQNRecurrentCommTrainer.
            communication_module (Type[BaseCommunicationModule], optional): module for
                enabling communication protocols between agents. Defaults to
                BroadcastedCommunication.
            executor_fn (Type[core.Executor], optional): executor type, e.g.
                feedforward or recurrent. Defaults to
                execution.MADQNFeedForwardExecutor.
            exploration_scheduler_fn (Type[ LinearExplorationScheduler ], optional):
                function specifying a decaying scheduler for epsilon exploration.
                See mava/systems/tf/madqn/system.py for details.
            replay_stabilisation_fn (Optional[Type[FingerPrintStabalisation]],
                optional): replay buffer stabilisaiton function, e.g. fingerprints.
                Defaults to None.
            num_executors (int, optional): number of executor processes to run in
                parallel. Defaults to 1.
            num_caches (int, optional): number of trainer node caches. Defaults to 0.
            environment_spec (mava_specs.MAEnvironmentSpec, optional): description of
                the action, observation spaces etc. for each agent in the system.
                Defaults to None.
            shared_weights (bool, optional): whether agents should share weights or not.
                When agent_net_keys are provided the value of shared_weights is ignored.
                Defaults to True.
            agent_net_keys: (dict, optional): specifies what network each agent uses.
                Defaults to {}.
            batch_size (int, optional): sample batch size for updates. Defaults to 256.
            prefetch_size (int, optional): size to prefetch from replay. Defaults to 4.
            min_replay_size (int, optional): minimum replay size before updating.
                Defaults to 1000.
            max_replay_size (int, optional): maximum replay size. Defaults to 1000000.
            samples_per_insert (Optional[float], optional): number of samples to take
                from replay for every insert that is made. Defaults to 4.0.
            n_step (int, optional): number of steps to include prior to boostrapping.
                Defaults to 5.
            sequence_length (int, optional): recurrent sequence rollout length.
                Defaults to 6.
            period (int, optional): consecutive starting points for overlapping
                rollouts across a sequence. Defaults to 20.
            max_gradient_norm (float, optional): maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            discount (float, optional): discount factor to use for TD updates.
                Defaults to 1.
            optimizer (Union[snt.Optimizer, Dict[str, snt.Optimizer]], optional):
                type of optimizer to use to update network parameters. Defaults to
                snt.optimizers.Adam( learning_rate=1e-4 ).
            target_update_period (int, optional): number of steps before target
                networks are updated. Defaults to 100.
            executor_variable_update_period (int, optional): number of steps before
                updating executor variables from the variable source. Defaults to 1000.
            max_executor_steps (int, optional): maximum number of steps and executor
                can in an episode. Defaults to None.
            checkpoint (bool, optional): whether to checkpoint models. Defaults to
                False.
            checkpoint_subpath (str, optional): subdirectory specifying where to store
                checkpoints. Defaults to "~/mava/".
            checkpoint_minute_interval (int): The number of minutes to wait between
                checkpoints.
            logger_config (Dict, optional): additional configuration settings for the
                logger factory. Defaults to {}.
            train_loop_fn (Callable, optional): function to instantiate a train loop.
                Defaults to ParallelEnvironmentLoop.
            eval_loop_fn (Callable, optional): function to instantiate an evaluation
                loop. Defaults to ParallelEnvironmentLoop.
            train_loop_fn_kwargs (Dict, optional): possible keyword arguments to send
                to the training loop. Defaults to {}.
            eval_loop_fn_kwargs (Dict, optional): possible keyword arguments to send to
                the evaluation loop. Defaults to {}.
            learning_rate_scheduler_fn: function/class that takes in a trainer step t
                and returns the current learning rate.
            seed: seed for reproducible sampling (for epsilon greedy action selection).
            evaluator_interval: An optional condition that is used to evaluate/test
                system performance after [evaluator_interval] condition has been met.
                If None, evaluation will happen at every timestep.
                E.g. to evaluate a system after every 100 executor episodes,
                evaluator_interval = {"executor_episodes": 100}.
        """

        super(DIAL, self).__init__(
            environment_factory=environment_factory,
            network_factory=network_factory,
            logger_factory=logger_factory,
            architecture=architecture,
            trainer_fn=trainer_fn,
            communication_module=communication_module,
            executor_fn=executor_fn,
            replay_stabilisation_fn=replay_stabilisation_fn,
            num_executors=num_executors,
            num_caches=num_caches,
            environment_spec=environment_spec,
            agent_net_keys=agent_net_keys,
            shared_weights=shared_weights,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
            min_replay_size=min_replay_size,
            max_replay_size=max_replay_size,
            samples_per_insert=samples_per_insert,
            n_step=n_step,
            sequence_length=sequence_length,
            period=period,
            discount=discount,
            optimizer=optimizer,
            target_update_period=target_update_period,
            executor_variable_update_period=executor_variable_update_period,
            max_executor_steps=max_executor_steps,
            checkpoint=checkpoint,
            checkpoint_subpath=checkpoint_subpath,
            checkpoint_minute_interval=checkpoint_minute_interval,
            logger_config=logger_config,
            train_loop_fn=train_loop_fn,
            eval_loop_fn=eval_loop_fn,
            train_loop_fn_kwargs=train_loop_fn_kwargs,
            eval_loop_fn_kwargs=eval_loop_fn_kwargs,
            exploration_scheduler_fn=exploration_scheduler_fn,
            learning_rate_scheduler_fn=learning_rate_scheduler_fn,
            seed=seed,
        )

        if issubclass(executor_fn, executors.RecurrentExecutor):
            extra_specs = self._get_extra_specs()
        else:
            extra_specs = {}
        self._checkpoint_minute_interval = checkpoint_minute_interval
        self._builder = builder.DIALBuilder(
            builder.DIALConfig(
                environment_spec=self._environment_spec,
                agent_net_keys=self._agent_net_keys,
                discount=discount,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                executor_variable_update_period=executor_variable_update_period,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                samples_per_insert=samples_per_insert,
                n_step=n_step,
                sequence_length=sequence_length,
                period=period,
                max_gradient_norm=max_gradient_norm,
                checkpoint=checkpoint,
                optimizer=optimizer,
                checkpoint_subpath=checkpoint_subpath,
                checkpoint_minute_interval=checkpoint_minute_interval,
                learning_rate_scheduler_fn=learning_rate_scheduler_fn,
                importance_sampling_exponent=importance_sampling_exponent,
                max_priority_weight=max_priority_weight,
                evaluator_interval=evaluator_interval,
            ),
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
            replay_stabilisation_fn=replay_stabilisation_fn,
        )

    def replay(self) -> Any:
        """Replay data storage.

        Returns:
            Any: replay data table built according the environment specification.
        """
        return self._builder.make_replay_tables(self._environment_spec)

    def executor(  # type: ignore[override]
        self,
        executor_id: str,
        replay: reverb.Client,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        trainer: Optional[training.MADQNRecurrentCommTrainer] = None,
    ) -> mava.ParallelEnvironmentLoop:
        """System executor

        Args:
            executor_id (str): id to identify the executor process for logging purposes.
            replay (reverb.Client): replay data table to push data to.
            variable_source (acme.VariableSource): variable server for updating
                network variables.
            counter (counting.Counter): step counter object.
            trainer (Optional[training.MADQNRecurrentCommTrainer], optional):
                system trainer. Defaults to None.

        Returns:
            mava.ParallelEnvironmentLoop: environment-executor loop instance.
        """

        return super().executor(
            executor_id=executor_id,
            replay=replay,
            variable_source=variable_source,
            counter=counter,
            trainer=trainer,
        )

    def evaluator(  # type: ignore[override]
        self,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        trainer: training.MADQNRecurrentCommTrainer,
    ) -> Any:
        """System evaluator (an executor process not connected to a dataset)

        Args:
            variable_source (acme.VariableSource): variable server for updating
                network variables.
            counter (counting.Counter): step counter object.
            trainer (Optional[training.MADQNRecurrentCommTrainer], optional):
                system trainer. Defaults to None.

        Returns:
            Any: environment-executor evaluation loop instance for evaluating the
                performance of a system.
        """

        return super().evaluator(
            variable_source=variable_source,
            counter=counter,
            trainer=trainer,
        )

    def build(self, name: str = "dial") -> Any:
        """Build the distributed system as a graph program.

        Args:
            name (str, optional): system name. Defaults to "dial".

        Returns:
            Any: graph program for distributed system training.
        """

        return super().build(name=name)
