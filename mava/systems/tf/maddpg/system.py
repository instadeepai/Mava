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

"""MADDPG system implementation."""

import functools
from typing import Any, Callable, Dict, List, Optional, Type, Union

import acme
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
from acme import specs as acme_specs
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava
from mava import core
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedQValueActorCritic
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf import savers as tf2_savers
from mava.systems.tf.maddpg import builder, training
from mava.systems.tf.maddpg.execution import MADDPGFeedForwardExecutor
from mava.utils import lp_utils
from mava.utils.loggers import MavaLogger, logger_utils
from mava.wrappers import DetailedPerAgentStatistics


class MADDPG:
    """MADDPG system."""

    def __init__(
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        logger_factory: Callable[[str], MavaLogger] = None,
        architecture: Type[
            DecentralisedQValueActorCritic
        ] = DecentralisedQValueActorCritic,
        trainer_fn: Union[
            Type[training.MADDPGBaseTrainer],
            Type[training.MADDPGBaseRecurrentTrainer],
        ] = training.MADDPGDecentralisedTrainer,
        executor_fn: Type[core.Executor] = MADDPGFeedForwardExecutor,
        num_executors: int = 1,
        num_caches: int = 0,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        discount: float = 0.99,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_averaging: bool = False,
        target_update_period: int = 100,
        target_update_rate: Optional[float] = None,
        executor_variable_update_period: int = 1000,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: Optional[float] = 32.0,
        policy_optimizer: Union[
            snt.Optimizer, Dict[str, snt.Optimizer]
        ] = snt.optimizers.Adam(learning_rate=1e-4),
        critic_optimizer: snt.Optimizer = snt.optimizers.Adam(learning_rate=1e-4),
        n_step: int = 5,
        sequence_length: int = 20,
        period: int = 20,
        sigma: float = 0.3,
        max_gradient_norm: float = None,
        max_executor_steps: int = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        logger_config: Dict = {},
        train_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        train_loop_fn_kwargs: Dict = {},
        eval_loop_fn_kwargs: Dict = {},
        connection_spec: Callable[[Dict[str, List[str]]], Dict[str, List[str]]] = None,
    ):
        """Initialise the system

        Args:
            environment_factory (Callable[[bool], dm_env.Environment]): function to
                instantiate an environment.
            network_factory (Callable[[acme_specs.BoundedArray],
                Dict[str, snt.Module]]): function to instantiate system networks.
            logger_factory (Callable[[str], MavaLogger], optional): function to
                instantiate a system logger. Defaults to None.
            architecture (Type[ DecentralisedQValueActorCritic ], optional):
                system architecture, e.g. decentralised or centralised. Defaults to
                DecentralisedQValueActorCritic.
            trainer_fn (Union[ Type[training.MADDPGBaseTrainer],
                Type[training.MADDPGBaseRecurrentTrainer], ], optional): training type
                associated with executor and architecture, e.g. centralised training.
                Defaults to training.MADDPGDecentralisedTrainer.
            executor_fn (Type[core.Executor], optional): executor type, e.g.
                feedforward or recurrent. Defaults to MADDPGFeedForwardExecutor.
            num_executors (int, optional): number of executor processes to run in
                parallel. Defaults to 1.
            num_caches (int, optional): number of trainer node caches. Defaults to 0.
            environment_spec (mava_specs.MAEnvironmentSpec, optional): description of
                the action, observation spaces etc. for each agent in the system.
                Defaults to None.
            shared_weights (bool, optional): whether agents should share weights or not.
                Defaults to True.
            discount (float, optional): discount factor to use for TD updates. Defaults
                to 0.99.
            batch_size (int, optional): sample batch size for updates. Defaults to 256.
            prefetch_size (int, optional): size to prefetch from replay. Defaults to 4.
            target_averaging (bool, optional): whether to use polyak averaging for
                target network updates. Defaults to False.
            target_update_period (int, optional): number of steps before target
                networks are updated. Defaults to 100.
            target_update_rate (Optional[float], optional): update rate when using
                averaging. Defaults toNone.
            executor_variable_update_period (int, optional): number of steps before
                updating executor variables from the variable source. Defaults to 1000.
            min_replay_size (int, optional): minimum replay size before updating.
                Defaults to 1000.
            max_replay_size (int, optional): maximum replay size. Defaults to 1000000.
            samples_per_insert (Optional[float], optional): number of samples to take
                from replay for every insert that is made. Defaults to 32.0.
            policy_optimizer (Union[ snt.Optimizer, Dict[str, snt.Optimizer] ],
                optional): optimizer(s) for updating policy networks. Defaults to
                snt.optimizers.Adam(learning_rate=1e-4).
            critic_optimizer (snt.Optimizer, optional): optimizer for updating critic
                networks. Defaults to snt.optimizers.Adam(learning_rate=1e-4).
            n_step (int, optional): number of steps to include prior to boostrapping.
                Defaults to 5.
            sequence_length (int, optional): recurrent sequence rollout length. Defaults
                to 20.
            period (int, optional): [consecutive starting points for overlapping
                rollouts across a sequence. Defaults to 20.
            sigma (float, optional): Gaussian sigma parameter. Defaults to 0.3.
            max_gradient_norm (float, optional): maximum allowed norm for gradients
                before clipping is applied. Defaults to None.
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
            connection_spec (Callable[[Dict[str, List[str]]], Dict[str, List[str]]],
                optional): network topology specification for networked system
                architectures. Defaults to None.
        """

        if not environment_spec:
            environment_spec = mava_specs.MAEnvironmentSpec(
                environment_factory(evaluation=False)  # type: ignore
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
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._logger_factory = logger_factory
        self._environment_spec = environment_spec
        self._shared_weights = shared_weights
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

        if connection_spec:
            self._connection_spec = connection_spec(  # type: ignore
                environment_spec.get_agents_by_type()
            )
        else:
            self._connection_spec = None  # type: ignore

        if issubclass(executor_fn, executors.RecurrentExecutor):
            extra_specs = self._get_extra_specs()
        else:
            extra_specs = {}

        self._builder = builder.MADDPGBuilder(
            builder.MADDPGConfig(
                environment_spec=environment_spec,
                shared_weights=shared_weights,
                discount=discount,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_averaging=target_averaging,
                target_update_period=target_update_period,
                target_update_rate=target_update_rate,
                executor_variable_update_period=executor_variable_update_period,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                samples_per_insert=samples_per_insert,
                n_step=n_step,
                sequence_length=sequence_length,
                period=period,
                sigma=sigma,
                max_gradient_norm=max_gradient_norm,
                checkpoint=checkpoint,
                policy_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
                checkpoint_subpath=checkpoint_subpath,
            ),
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
        )

    def _get_extra_specs(self) -> Any:
        """helper to establish specs for extra information

        Returns:
            Dict[str, Any]: dictionary containing extra specs
        """

        agents = self._environment_spec.get_agent_ids()
        core_state_specs = {}
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec
        )
        for agent in agents:
            agent_type = agent.split("_")[0]
            core_state_specs[agent] = (
                tf2_utils.squeeze_batch_dim(
                    networks["policies"][agent_type].initial_state(1)
                ),
            )
        return {"core_states": core_state_specs}

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
            Any: step counter object.
        """

        if checkpoint:
            return tf2_savers.CheckpointingRunner(
                counting.Counter(),
                time_delta_minutes=15,
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

        return lp_utils.StepsLimiter(counter, self._max_executor_steps)

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
            environment_spec=self._environment_spec, shared_weights=self._shared_weights
        )

        # create logger
        trainer_logger_config = {}
        if self._logger_config and "trainer" in self._logger_config:
            trainer_logger_config = self._logger_config["trainer"]
        trainer_logger = self._logger_factory(  # type: ignore
            "trainer", **trainer_logger_config
        )

        # Create system architecture with target networks.
        adder_env_spec = self._builder.convert_discrete_to_bounded(
            self._environment_spec
        )

        # architecture args
        architecture_config = {
            "environment_spec": adder_env_spec,
            "observation_networks": networks["observations"],
            "policy_networks": networks["policies"],
            "critic_networks": networks["critics"],
            "shared_weights": self._shared_weights,
        }
        if self._connection_spec:
            architecture_config["network_spec"] = self._connection_spec

        system_networks = self._architecture(**architecture_config).create_system()

        dataset = self._builder.make_dataset_iterator(replay)
        counter = counting.Counter(counter, "trainer")

        return self._builder.make_trainer(
            networks=system_networks,
            dataset=dataset,
            counter=counter,
            logger=trainer_logger,
            connection_spec=self._connection_spec,
        )

    def executor(
        self,
        executor_id: str,
        replay: reverb.Client,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
    ) -> mava.ParallelEnvironmentLoop:
        """System executor

        Args:
            executor_id (str): id to identify the executor process for logging purposes.
            replay (reverb.Client): replay data table to push data to.
            variable_source (acme.VariableSource): variable server for updating
                network variables.
            counter (counting.Counter): step counter object.

        Returns:
            mava.ParallelEnvironmentLoop: environment-executor loop instance.
        """

        # Create the behavior policy.
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec, shared_weights=self._shared_weights
        )

        # architecture args
        architecture_config = {
            "environment_spec": self._environment_spec,
            "observation_networks": networks["observations"],
            "policy_networks": networks["policies"],
            "critic_networks": networks["critics"],
            "shared_weights": self._shared_weights,
        }
        if self._connection_spec:
            architecture_config["network_spec"] = self._connection_spec

        # Create system architecture with target networks.
        system = self._architecture(**architecture_config)

        # create variables
        _ = system.create_system()

        # behaviour policy networks (obs net + policy head)
        behaviour_policy_networks = system.create_behaviour_policy()

        # Create the executor.
        executor = self._builder.make_executor(
            policy_networks=behaviour_policy_networks,
            adder=self._builder.make_adder(replay),
            variable_source=variable_source,
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
        logger: loggers.Logger = None,
    ) -> Any:
        """System evaluator (an executor process not connected to a dataset)

        Args:
            variable_source (acme.VariableSource): variable server for updating
                network variables.
            counter (counting.Counter): step counter object.
            logger (loggers.Logger, optional): logger object. Defaults to None.

        Returns:
            Any: environment-executor evaluation loop instance for evaluating the
                performance of a system.
        """

        # Create the behavior policy.
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec, shared_weights=self._shared_weights
        )

        # architecture args
        architecture_config = {
            "environment_spec": self._environment_spec,
            "observation_networks": networks["observations"],
            "policy_networks": networks["policies"],
            "critic_networks": networks["critics"],
            "shared_weights": self._shared_weights,
        }
        if self._connection_spec:
            architecture_config["network_spec"] = self._connection_spec

        # Create system architecture with target networks.
        system = self._architecture(**architecture_config)

        # create variables
        _ = system.create_system()

        # behaviour policy networks (obs net + policy head)
        behaviour_policy_networks = system.create_behaviour_policy()

        # Create the agent.
        executor = self._builder.make_executor(
            policy_networks=behaviour_policy_networks,
            variable_source=variable_source,
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

    def build(self, name: str = "maddpg") -> Any:
        """Build the distributed system as a graph program.

        Args:
            name (str, optional): system name. Defaults to "maddpg".

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
            program.add_node(lp.CourierNode(self.evaluator, trainer, counter))

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
                    lp.CourierNode(self.executor, executor_id, replay, source, counter)
                )

        return program
