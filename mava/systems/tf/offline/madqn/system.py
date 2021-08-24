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

"""Offline MADQN system implementation."""

import functools
from mava.systems.tf.madqn.system import MADQN
from typing import Any, Callable, Dict, Optional, Type, Union

import acme
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
from acme import specs as acme_specs
from acme.tf import utils as tf2_utils
from acme.utils import counting

import mava
from mava import core
from mava.adders import reverb as reverb_adders
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedValueActor
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.components.tf.modules.exploration import LinearExplorationScheduler, exploration_scheduling
from mava.components.tf.modules.stabilising import FingerPrintStabalisation
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf import savers as tf2_savers
from mava.systems.tf.offline.madqn import builder, execution, training
from mava.utils import lp_utils
from mava.utils.loggers import MavaLogger, logger_utils
from mava.wrappers import DetailedPerAgentStatistics

from mava.systems.tf.madqn import MADQN


class OfflineMADQN(MADQN):
    """Offline MADQN system."""

    def __init__(
        self,
        reverb_checkpoint_dir: str,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[DecentralisedValueActor] = DecentralisedValueActor,
        trainer_fn: Type[training.OfflineMADQNTrainer] = training.OfflineMADQNTrainer,
        communication_module: Type[BaseCommunicationModule] = None,
        executor_fn: Type[core.Executor] = execution.MADQNFeedForwardExecutor,
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
        samples_per_insert: Optional[float] = None,
        n_step: int = 5,
        sequence_length: int = 20,
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
        logger_config: Dict = {},
        train_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        train_loop_fn_kwargs: Dict = {},
        eval_loop_fn_kwargs: Dict = {},
        replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
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
            trainer_fn (Union[ Type[training.MADQNTrainer],
                Type[training.MADQNRecurrentTrainer] ], optional): training type
                associated with executor and architecture, e.g. centralised training.
                Defaults to training.MADQNTrainer.
            communication_module (Type[BaseCommunicationModule], optional):
                module for enabling communication protocols between agents. Defaults to
                None.
            executor_fn (Type[core.Executor], optional): executor type, e.g.
                feedforward or recurrent. Defaults to
                execution.MADQNFeedForwardExecutor.
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
            period (int, optional): consecutive starting points for overlapping
                rollouts across a sequence. Defaults to 20.
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
        """

        if not environment_spec:
            environment_spec = mava_specs.MAEnvironmentSpec(
                environment_factory(evaluation=False)  # type:ignore
            )

        # Set default logger if no logger provided
        if not logger_factory:
            logger_factory = functools.partial(
                logger_utils.make_logger,
                directory="~/mava",
                to_terminal=True,
                time_delta=10,
            )

        # Store useful variables for later
        self._architecture = architecture
        self._communication_module_fn = communication_module
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._logger_factory = logger_factory
        self._environment_spec = environment_spec
        self._num_exectors = num_executors
        self._num_caches = num_caches
        self._max_executor_steps = max_executor_steps
        self._checkpoint_subpath = checkpoint_subpath
        self._checkpoint = checkpoint
        self._logger_config = logger_config
        self._train_loop_fn = train_loop_fn
        self._train_loop_fn_kwargs = train_loop_fn_kwargs
        self._eval_loop_fn = eval_loop_fn
        self._eval_loop_fn_kwargs = eval_loop_fn_kwargs

        # Setup agent networks
        self._agent_net_keys = agent_net_keys
        if not agent_net_keys:
            agents = environment_spec.get_agent_ids()
            self._agent_net_keys = {
                agent: agent.split("_")[0] if shared_weights else agent
                for agent in agents
            }

        # Extras spec is empty because system is not recurrent.
        extra_specs = {}

        # Assertions to check for incompatible components
        # RateLimiter can not be a SampleToInsert RateLimiter 
        assert samples_per_insert is None
        # Recurrent MADQN not supported.
        assert not issubclass(executor_fn, executors.RecurrentExecutor)

        # Make the builder
        self._builder = builder.OfflineMADQNBuilder(
            builder.OfflineMADQNConfig(
                environment_spec=environment_spec,
                epsilon_min=epsilon_min,
                exploration_scheduler=
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
                max_gradient_norm=max_gradient_norm,
                checkpoint=checkpoint,
                optimizer=optimizer,
                checkpoint_subpath=checkpoint_subpath,
                reverb_checkpoint_dir=reverb_checkpoint_dir,
                replay_table_name=replay_table_name
            ),
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
            exploration_scheduler_fn=exploration_scheduler_fn,
            replay_stabilisation_fn=replay_stabilisation_fn,
        )

    def reverb_checkpoint(self):
        """Reverb checkpoint factory.

        Returns:
            (reverb.checkpointers.DefaultCheckpointer): A reverb checkpointer to
                restore offline dataset.
        """
        return self._builder.make_reverb_checkpointer()

    def build(self, name: str = "madqn") -> Any:
        """Build the distributed system as a graph program.

        Args:
            name (str, optional): system name. Defaults to "madqn".

        Returns:
            Any: graph program for distributed system training.
        """

        """Build the distributed system as a graph program.

        Args:
            name (str, optional): system name. Defaults to "madqn".

        Returns:
            Any: graph program for distributed system training.
        """

        program = lp.Program(name=name)

        with program.group("replay"):
            replay = program.add_node(lp.ReverbNode(self.replay, self.reverb_checkpoint))

        with program.group("counter"):
            counter = program.add_node(lp.CourierNode(self.counter, self._checkpoint))

        if self._max_executor_steps:
            with program.group("coordinator"):
                _ = program.add_node(lp.CourierNode(self.coordinator, counter))

        with program.group("trainer"):
            trainer = program.add_node(lp.CourierNode(self.trainer, replay, counter))

        with program.group("evaluator"):
            program.add_node(lp.CourierNode(self.evaluator, trainer, counter, trainer))

        if not self._num_caches:
            # Use the trainer as a single variable source.
            sources = [trainer]
        else:
            with program.group("cacher"):
                # Create a set of trainer caches.
                sources = []
                for _ in range(self._num_caches):
                    cacher = program.add_node(
                        lp.CacherNode(
                            trainer, refresh_interval_ms=2000, stale_after_ms=4000
                        )
                    )
                    sources.append(cacher)

        return program