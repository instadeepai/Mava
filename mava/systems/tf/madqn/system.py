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

"""Defines the MADQN system class."""

import copy
from typing import Any, Callable, Dict, Optional, Tuple, Type

import acme
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
import tensorflow as tf
from acme import specs as acme_specs
from acme.utils import counting, loggers

import mava
from mava import core
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedValueActor
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import savers as tf2_savers
from mava.systems.tf.madqn import builder, execution, training
from mava.utils import lp_utils
from mava.utils.loggers.base import MavaLogger
from mava.wrappers import DetailedPerAgentStatistics


class MADQN:
    """Program definition for MADQN."""

    def __init__(
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        architecture: Type[DecentralisedValueActor] = DecentralisedValueActor,
        trainer_fn: Type[training.MADQNTrainer] = training.MADQNTrainer,
        executor_fn: Type[core.Executor] = execution.MADQNFeedForwardExecutor,
        num_executors: int = 1,
        num_caches: int = 0,
        log_info: Tuple = None,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        epsilon: float = tf.Variable(1.0, trainable=False),
        batch_size: int = 256,
        prefetch_size: int = 4,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: Optional[float] = 32.0,
        n_step: int = 5,
        clipping: bool = True,
        discount: float = 0.99,
        policy_optimizer: snt.Optimizer = snt.optimizers.Adam(learning_rate=1e-4),
        target_update_period: int = 100,
        executor_variable_update_period: int = 1000,
        log_every: float = 10.0,
        max_executor_steps: int = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        trainer_logger: MavaLogger = None,
        exec_logger: MavaLogger = None,
        eval_logger: MavaLogger = None,
    ):

        if not environment_spec:
            environment_spec = mava_specs.MAEnvironmentSpec(
                environment_factory(evaluation=False)  # type: ignore
            )

        self._architecture = architecture
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._log_info = log_info
        self._environment_spec = environment_spec
        self._shared_weights = shared_weights
        self._num_exectors = num_executors
        self._num_caches = num_caches
        self._max_executor_steps = max_executor_steps
        self._log_every = log_every
        self._checkpoint_subpath = checkpoint_subpath
        self._checkpoint = checkpoint
        self._trainer_logger = trainer_logger
        self._exec_logger = exec_logger
        self._eval_logger = eval_logger

        self._builder = builder.MADQNBuilder(
            builder.MADQNConfig(
                environment_spec=environment_spec,
                epsilon=epsilon,
                shared_weights=shared_weights,
                discount=discount,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                executor_variable_update_period=executor_variable_update_period,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                samples_per_insert=samples_per_insert,
                n_step=n_step,
                clipping=clipping,
                checkpoint=checkpoint,
                policy_optimizer=policy_optimizer,
                checkpoint_subpath=checkpoint_subpath,
            ),
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
        )

    def replay(self) -> Any:
        """The replay storage."""
        return self._builder.make_replay_tables(self._environment_spec)

    def counter(self) -> Any:
        return tf2_savers.CheckpointingRunner(
            counting.Counter(),
            time_delta_minutes=15,
            directory=self._checkpoint_subpath,
            subdirectory="counter",
        )

    def coordinator(self, counter: counting.Counter) -> Any:
        return lp_utils.StepsLimiter(counter, self._max_executor_steps)  # type: ignore

    def trainer(
        self,
        replay: reverb.Client,
        counter: counting.Counter,
    ) -> mava.core.Trainer:
        """The Trainer part of the system."""

        # Create the networks to optimize (online)
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec
        )

        # Create system architecture with target networks.
        system_networks = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            shared_weights=self._shared_weights,
        ).create_system()

        dataset = self._builder.make_dataset_iterator(replay)
        counter = counting.Counter(counter, "trainer")

        return self._builder.make_trainer(
            networks=system_networks,
            dataset=dataset,
            counter=counter,
            logger=self._trainer_logger,
        )

    def executor(
        self,
        executor_id: str,
        replay: reverb.Client,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
    ) -> mava.ParallelEnvironmentLoop:
        """The executor process."""

        # Create the behavior policy.
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec
        )

        # Create system architecture with target networks.
        executor_networks = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            shared_weights=self._shared_weights,
        ).create_system()

        # Create the executor.
        executor = self._builder.make_executor(
            q_networks=executor_networks["values"],
            action_selectors=networks["action_selectors"],
            adder=self._builder.make_adder(replay),
        )

        # TODO (Arnu): figure out why factory function are giving type errors
        # Create the environment.
        environment = self._environment_factory(evaluation=False)  # type: ignore

        # Create logger and counter; actors will not spam bigtable.
        counter = counting.Counter(counter, "executor")

        # Update label to include exec id
        exec_logger = None
        if self._exec_logger:
            exec_logger = copy.deepcopy(self._exec_logger)
            exec_logger._label = f"{exec_logger._label}_{executor_id}"  # type: ignore

        # Create the loop to connect environment and executor.
        train_loop = ParallelEnvironmentLoop(
            environment, executor, counter=counter, logger=exec_logger
        )

        train_loop = DetailedPerAgentStatistics(train_loop)

        return train_loop

    def evaluator(
        self,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        logger: loggers.Logger = None,
    ) -> Any:
        """The evaluation process."""

        # Create the behavior policy.
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec
        )

        # Create system architecture with target networks.
        executor_networks = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            shared_weights=self._shared_weights,
        ).create_system()

        # Create the agent.
        executor = self._builder.make_executor(
            q_networks=executor_networks["values"],
            action_selectors=networks["action_selectors"],
        )

        # Make the environment.
        environment = self._environment_factory(evaluation=True)  # type: ignore

        # Create logger and counter.
        counter = counting.Counter(counter, "evaluator")

        # Create the run loop and return it.
        # Create the loop to connect environment and executor.
        eval_loop = ParallelEnvironmentLoop(
            environment, executor, counter=counter, logger=self._eval_logger
        )

        eval_loop = DetailedPerAgentStatistics(eval_loop)
        return eval_loop

    def build(self, name: str = "madqn") -> Any:
        """Build the distributed system topology."""
        program = lp.Program(name=name)

        with program.group("replay"):
            replay = program.add_node(lp.ReverbNode(self.replay))

        with program.group("counter"):
            counter = program.add_node(lp.CourierNode(self.counter))

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
