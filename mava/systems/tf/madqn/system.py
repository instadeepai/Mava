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

"""MADQN system implementation."""

import functools
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
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedValueActor
from mava.components.tf.modules.communication import BaseCommunicationModule
from mava.components.tf.modules.exploration import LinearExplorationScheduler
from mava.components.tf.modules.stabilising import FingerPrintStabalisation
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf import savers as tf2_savers
from mava.systems.tf.madqn import builder, execution, training
from mava.utils import lp_utils
from mava.utils.loggers import MavaLogger, logger_utils
from mava.wrappers import DetailedPerAgentStatistics


class MADQN:
    """MADQN system."""

    def __init__(
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[DecentralisedValueActor] = DecentralisedValueActor,
        trainer_fn: Union[
            Type[training.MADQNTrainer], Type[training.MADQNRecurrentTrainer]
        ] = training.MADQNTrainer,
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
            importance_sampling_exponent (float, optional): value of importance sampling
                exponent (usually around 0.2). If None, importance sampling is not used.
            max_priority_weight (float): Required if importance_sampling_exponent
                is not None. Defaults to 0.9. Used to scale the maximum priority of
                reverb samples.
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
        """

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

        self._architecture = architecture
        self._communication_module_fn = communication_module
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._logger_factory = logger_factory
        self._environment_spec = environment_spec
        # Setup agent networks
        self._agent_net_keys = agent_net_keys
        if not agent_net_keys:
            agents = environment_spec.get_agent_ids()
            self._agent_net_keys = {
                agent: agent.split("_")[0] if shared_weights else agent
                for agent in agents
            }
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
        self._checkpoint_minute_interval = checkpoint_minute_interval

        if issubclass(executor_fn, executors.RecurrentExecutor):
            extra_specs = self._get_extra_specs()
        else:
            extra_specs = {}

        self._builder = builder.MADQNBuilder(
            builder.MADQNConfig(
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
        )

    def _get_extra_specs(self) -> Any:
        """helper to establish specs for extra information

        Returns:
            Dict[str, Any]: dictionary containing extra specs
        """

        agents = self._environment_spec.get_agent_ids()
        core_state_specs = {}
        core_message_specs = {}

        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
        )
        for agent in agents:
            agent_type = agent.split("_")[0]
            core_state_specs[agent] = (
                tf2_utils.squeeze_batch_dim(
                    networks["q_networks"][agent_type].initial_state(1)
                ),
            )
            if self._communication_module_fn is not None:
                core_message_specs[agent] = (
                    tf2_utils.squeeze_batch_dim(
                        networks["q_networks"][agent_type].initial_message(1)
                    ),
                )

        extras = {
            "core_states": core_state_specs,
            "core_messages": core_message_specs,
        }
        return extras

    def replay(self) -> Any:
        """Replay data storage.

        Returns:
            Any: replay data table built according the environment specification.
        """

        return self._builder.make_replay_tables(self._environment_spec)

    def counter(self, checkpoint: bool) -> Any:
        """Step counter

        Args:
            checkpoint (bool): whether to checkpoint the counter.

        Returns:
            Any: checkpointing object logging steps in a counter subdirectory.
        """

        if checkpoint:
            return tf2_savers.CheckpointingRunner(
                counting.Counter(),
                time_delta_minutes=self._checkpoint_minute_interval,
                directory=self._checkpoint_subpath,
                subdirectory="counter",
            )
        else:
            return counting.Counter()

    def coordinator(self, counter: counting.Counter) -> Any:
        """Coordination helper for a distributed program

        Args:
            counter (counting.Counter): step counter object.

        Returns:
            Any: step limiter object.
        """

        return lp_utils.StepsLimiter(counter, self._max_executor_steps)  # type: ignore

    def trainer(
        self,
        replay: reverb.Client,
        counter: counting.Counter,
    ) -> mava.core.Trainer:
        """System trainer

        Args:
            replay (reverb.Client): replay data table to pull data from.
            counter (counting.Counter): step counter object.

        Returns:
            mava.core.Trainer: system trainer.
        """

        # Create the networks to optimize (online)
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
        )

        # Create system architecture with target networks.
        architecture = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            agent_net_keys=self._agent_net_keys,
        )

        if self._builder._replay_stabiliser_fn is not None:
            architecture = self._builder._replay_stabiliser_fn(  # type: ignore
                architecture
            )

        communication_module = None
        if self._communication_module_fn is not None:
            communication_module = self._communication_module_fn(
                architecture=architecture,
                shared=True,
                channel_size=1,
                channel_noise=0,
            )
            system_networks = communication_module.create_system()
        else:
            system_networks = architecture.create_system()

        # create logger
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
            replay_client=replay,
            counter=counter,
            communication_module=communication_module,
            logger=trainer_logger,
        )

    def executor(
        self,
        executor_id: str,
        replay: reverb.Client,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        trainer: Optional[training.MADQNTrainer] = None,
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

        # Create the behavior policy.
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
        )

        # Create system architecture with target networks.
        architecture = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            agent_net_keys=self._agent_net_keys,
        )

        if self._builder._replay_stabiliser_fn is not None:
            architecture = self._builder._replay_stabiliser_fn(  # type: ignore
                architecture
            )

        communication_module = None
        if self._communication_module_fn is not None:
            communication_module = self._communication_module_fn(
                architecture=architecture,
                shared=True,
                channel_size=1,
                channel_noise=0,
            )
            system_networks = communication_module.create_system()
        else:
            system_networks = architecture.create_system()

        # Create the executor.
        executor = self._builder.make_executor(
            q_networks=system_networks["values"],
            action_selectors=networks["action_selectors"],
            communication_module=communication_module,
            adder=self._builder.make_adder(replay),
            variable_source=variable_source,
            trainer=trainer,
        )

        # TODO (Arnu): figure out why factory function are giving type errors
        # Create the environment.
        environment = self._environment_factory(evaluation=False)  # type: ignore

        # Create logger and counter; actors will not spam bigtable.
        counter = counting.Counter(counter, "executor")

        # Create executor logger
        executor_logger_config = {}
        if self._logger_config and "executor" in self._logger_config:
            executor_logger_config = self._logger_config["executor"]
        exec_logger = self._logger_factory(  # type: ignore
            f"executor_{executor_id}", **executor_logger_config
        )

        # Create the loop to connect environment and executor.
        train_loop = self._train_loop_fn(
            environment,
            executor,
            counter=counter,
            logger=exec_logger,
            **self._train_loop_fn_kwargs,
        )

        train_loop = DetailedPerAgentStatistics(train_loop)

        return train_loop

    def evaluator(
        self,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        trainer: training.MADQNTrainer,
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

        # Create the behavior policy.
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec,
            agent_net_keys=self._agent_net_keys,
        )

        # Create system architecture with target networks.
        architecture = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            agent_net_keys=self._agent_net_keys,
        )

        if self._builder._replay_stabiliser_fn is not None:
            architecture = self._builder._replay_stabiliser_fn(  # type: ignore
                architecture
            )

        communication_module = None
        if self._communication_module_fn is not None:
            communication_module = self._communication_module_fn(
                architecture=architecture,
                shared=True,
                channel_size=1,
                channel_noise=0,
            )
            system_networks = communication_module.create_system()
        else:
            system_networks = architecture.create_system()

        # Create the agent.
        executor = self._builder.make_executor(
            q_networks=system_networks["values"],
            action_selectors=networks["action_selectors"],
            variable_source=variable_source,
            communication_module=communication_module,
            trainer=trainer,
            evaluator=True,
        )

        # Make the environment.
        environment = self._environment_factory(evaluation=True)  # type: ignore

        # Create logger and counter.
        counter = counting.Counter(counter, "evaluator")
        evaluator_logger_config = {}
        if self._logger_config and "evaluator" in self._logger_config:
            evaluator_logger_config = self._logger_config["evaluator"]
        eval_logger = self._logger_factory(  # type: ignore
            "evaluator", **evaluator_logger_config
        )

        # Create the run loop and return it.
        # Create the loop to connect environment and executor.
        eval_loop = self._eval_loop_fn(
            environment,
            executor,
            counter=counter,
            logger=eval_logger,
            **self._eval_loop_fn_kwargs,
        )

        eval_loop = DetailedPerAgentStatistics(eval_loop)
        return eval_loop

    def build(self, name: str = "madqn") -> Any:
        """Build the distributed system as a graph program.

        Args:
            name (str, optional): system name. Defaults to "madqn".

        Returns:
            Any: graph program for distributed system training.
        """

        program = lp.Program(name=name)

        with program.group("replay"):
            replay = program.add_node(lp.ReverbNode(self.replay))

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

        with program.group("executor"):
            # Add executors which pull round-robin from our variable sources.
            for executor_id in range(self._num_exectors):
                source = sources[executor_id % len(sources)]
                program.add_node(
                    lp.CourierNode(
                        self.executor,
                        executor_id,
                        replay,
                        source,
                        counter,
                        trainer,
                    )
                )

        return program
