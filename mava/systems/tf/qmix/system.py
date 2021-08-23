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

"""QMIX system implementation."""

import functools
from typing import Any, Callable, Dict, Optional, Type, Union

import dm_env
import reverb
import sonnet as snt
from acme import specs as acme_specs
from acme.utils import counting

import mava
from mava import core
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedValueActor
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.components.tf.modules.exploration import LinearExplorationScheduler
from mava.components.tf.modules.mixing import MonotonicMixing
from mava.components.tf.modules.stabilising import FingerPrintStabalisation
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf.madqn.system import MADQN
from mava.systems.tf.qmix import builder, execution, training
from mava.utils.loggers import MavaLogger, logger_utils


# TODO Implement recurrent QMIX
class QMIX(MADQN):
    """QMIX system."""

    def __init__(
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[DecentralisedValueActor] = DecentralisedValueActor,
        trainer_fn: Type[training.QMIXTrainer] = training.QMIXTrainer,
        executor_fn: Type[core.Executor] = execution.QMIXFeedForwardExecutor,
        mixer: Type[MonotonicMixing] = MonotonicMixing,
        communication_module: Type[BaseCommunicationModule] = None,
        exploration_scheduler_fn: Type[
            LinearExplorationScheduler
        ] = LinearExplorationScheduler,
        replay_stabilisation_fn: Optional[Type[FingerPrintStabalisation]] = None,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 1e-4,
        num_executors: int = 1,
        num_caches: int = 0,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        agent_net_keys: Dict[str, str] = {},
        batch_size: int = 256,
        prefetch_size: int = 4,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: Optional[float] = 32.0,
        n_step: int = 5,
        sequence_length: int = 20,
        importance_sampling_exponent: Optional[float] = None,
        max_priority_weight: float = 0.9,
        period: int = 20,
        max_gradient_norm: float = None,
        discount: float = 0.99,
        optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]] = snt.optimizers.Adam(
            learning_rate=1e-4
        ),
        target_update_period: int = 100,
        executor_variable_update_period: int = 1000,
        max_executor_steps: int = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        checkpoint_minute_interval: int = 5,
        logger_config: Dict = {},
        train_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        train_loop_fn_kwargs: Dict = {},
        eval_loop_fn_kwargs: Dict = {},
        qmix_hidden_dim: int = 32,
        num_hypernet_layers: int = 1,
        hypernet_hidden_dim: int = 32,
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
            trainer_fn (Type[training.QMIXTrainer], optional): training type associated
                with executor and architecture, e.g. centralised training. Defaults to
                training.QMIXTrainer.
            executor_fn (Type[core.Executor], optional): executor type, e.g.
                feedforward or recurrent. Defaults to execution.QMIXFeedForwardExecutor.
            mixer (Type[MonotonicMixing], optional): mixer module type, e.g. additive or
                monotonic mixing. Defaults to MonotonicMixing.
            communication_module (Type[BaseCommunicationModule], optional):
                module for enabling communication protocols between agents. Defaults to
                None.
            exploration_scheduler_fn (Type[ LinearExplorationScheduler ], optional):
                function specifying a decaying scheduler for epsilon exploration.
                Defaults to LinearExplorationScheduler.
            replay_stabilisation_fn (Optional[Type[FingerPrintStabalisation]],
                optional): replay buffer stabilisaiton function, e.g. fingerprints.
                Defaults to None.
            epsilon_min (float, optional): final minimum epsilon value at the end of a
                decaying schedule. Defaults to 0.05.
            epsilon_decay (float, optional): epsilon decay rate. Defaults to 1e-4.
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
                from replay for every insert that is made. Defaults to 32.0.
            n_step (int, optional): number of steps to include prior to boostrapping.
                Defaults to 5.
            sequence_length (int, optional): recurrent sequence rollout length. Defaults
                to 20.
            period (int, optional): The period with which we add sequences. See `period`
                in `acme.SequenceAdder.period` for more info. Defaults to 20.
            max_gradient_norm (float, optional): maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
            discount (float, optional): discount factor to use for TD updates. Defaults
                to 0.99.
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
            qmix_hidden_dim (int, optional): mixing network hidden dimension. Defaults
                to 32.
            num_hypernet_layers (int, optional): number of layers for hypernetwork.
                Defaults to 1.
            hypernet_hidden_dim (int, optional): hypernetwork hidden dimension. Defaults
                to 32.
        """

        self._mixer = mixer
        self._qmix_hidden_dim = qmix_hidden_dim
        self._num_hypernet_layers = num_hypernet_layers
        self._hypernet_hidden_dim = hypernet_hidden_dim

        if not environment_spec:
            environment_spec = mava_specs.MAEnvironmentSpec(
                environment_factory(evaluation=False)  # type:ignore
            )

        # set default logger if no logger provided
        if not logger_factory:
            logger_factory = functools.partial(
                logger_utils.make_logger,
                directory="~/mava",
                to_terminal=True,
                time_delta=10,
            )

        super(QMIX, self).__init__(
            environment_factory=environment_factory,
            network_factory=network_factory,
            logger_factory=logger_factory,
            architecture=architecture,
            trainer_fn=trainer_fn,
            communication_module=communication_module,
            executor_fn=executor_fn,
            exploration_scheduler_fn=exploration_scheduler_fn,
            replay_stabilisation_fn=replay_stabilisation_fn,
            epsilon_min=epsilon_min,
            epsilon_decay=epsilon_decay,
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
        )

        if issubclass(executor_fn, executors.RecurrentExecutor):
            extra_specs = self._get_extra_specs()
        else:
            extra_specs = {}

        self._builder = builder.QMIXBuilder(
            builder.QMIXConfig(
                environment_spec=environment_spec,
                epsilon_min=epsilon_min,
                epsilon_decay=epsilon_decay,
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
                importance_sampling_exponent=importance_sampling_exponent,
                max_priority_weight=max_priority_weight,
                period=period,
                max_gradient_norm=max_gradient_norm,
                checkpoint=checkpoint,
                optimizer=optimizer,
                checkpoint_subpath=checkpoint_subpath,
                checkpoint_minute_interval=checkpoint_minute_interval,
            ),
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
            exploration_scheduler_fn=exploration_scheduler_fn,
            replay_stabilisation_fn=replay_stabilisation_fn,
            mixer=mixer,
        )

    def trainer(
        self,
        replay: reverb.Client,
        counter: counting.Counter,
    ) -> mava.core.Trainer:
        """System trainer

        Args:
            replay (reverb.Client): replay data table to pull data from.
            counter (counting.Counter): step counter object.

        Raises:
            Exception: no option for recurrence (yet).

        Returns:
            mava.core.Trainer: system trainer.
        """

        # Create the networks to optimize (online)
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
        )

        # Create system architecture
        architecture = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            agent_net_keys=self._agent_net_keys,
        )

        # Fingerprint module
        if self._builder._replay_stabiliser_fn is not None:
            architecture = self._builder._replay_stabiliser_fn(  # type: ignore
                architecture
            )

        # Communication module
        # NOTE: this is currently not expected to work with qmix
        # since we do not have a recurrent version.
        if self._communication_module_fn is not None:
            raise Exception(
                "QMIX currently does not support recurrence and \
                therefore cannot use a communication module."
            )

        # Mixing module
        system_networks = self._mixer(
            environment_spec=self._environment_spec,
            architecture=architecture,
            num_hypernet_layers=self._num_hypernet_layers,
            qmix_hidden_dim=self._qmix_hidden_dim,
            hypernet_hidden_dim=self._hypernet_hidden_dim,
        ).create_system()

        # Create logger
        trainer_logger_config = {}
        if self._logger_config and "trainer" in self._logger_config:
            trainer_logger_config = self._logger_config["trainer"]
        trainer_logger = self._logger_factory(  # type: ignore
            "trainer", **trainer_logger_config
        )

        dataset = self._builder.make_dataset_iterator(replay)
        counter = counting.Counter(counter, "trainer")

        return self._builder.make_trainer(
            networks=system_networks,
            dataset=dataset,
            counter=counter,
            communication_module=None,
            logger=trainer_logger,
        )

    def build(self, name: str = "qmix") -> Any:
        """Build the distributed system as a graph program.

        Args:
            name (str, optional): system name. Defaults to "qmix".

        Returns:
            Any: graph program for distributed system training.
        """
        return super().build(name=name)
