# python3
# Copyright 2018 DeepMind Technologies Limited. All rights reserved.
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

"""Defines the distributed MADQN system class."""

from typing import Any, Callable, Dict, Optional, Type

import acme
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
from acme import specs as acme_specs
from acme.tf import savers as tf2_savers
from acme.utils import counting, loggers

import mava
from mava import core
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedValueActor
from mava.systems.tf import executors
from mava.systems.tf.madqn import system, training
from mava.utils import lp_utils


class DistributedMADQN:
    """Program definition for MADQN."""

    def __init__(
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        architecture: Type[DecentralisedValueActor] = DecentralisedValueActor,
        trainer_fn: Type[training.IDQNTrainer] = training.IDQNTrainer,
        executor_fn: Type[core.Executor] = executors.FeedForwardExecutor,
        num_executors: int = 1,
        num_caches: int = 0,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        epsilon: float = 0.05,
        batch_size: int = 256,
        prefetch_size: int = 4,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: Optional[float] = 32.0,
        n_step: int = 5,
        clipping: bool = True,
        discount: float = 0.99,
        policy_optimizer: snt.Optimizer = None,
        target_update_period: int = 100,
        max_executor_steps: int = 10000,
        log_every: float = 10.0,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = False,
    ):

        if not environment_spec:
            environment_spec = mava_specs.MAEnvironmentSpec(environment_factory(False))

        self._architecture = architecture
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._environment_spec = environment_spec
        self._shared_weights = shared_weights
        self._num_exectors = num_executors
        self._num_caches = num_caches
        self._max_executor_steps = max_executor_steps
        self._log_every = log_every

        self._builder = system.MADQNBuilder(
            system.MADQNConfig(
                environment_spec=environment_spec,
                epsilon=epsilon,
                shared_weights=shared_weights,
                discount=discount,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                policy_optimizer=policy_optimizer,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                samples_per_insert=samples_per_insert,
                n_step=n_step,
                clipping=clipping,
                counter=counter,
                logger=logger,
                checkpoint=checkpoint,
            ),
            trainer_fn=trainer_fn,
            executer_fn=executor_fn,
        )

    def replay(self) -> Any:
        """The replay storage."""
        return self._builder.make_replay_tables(self._environment_spec)

    def counter(self) -> Any:
        return tf2_savers.CheckpointingRunner(
            counting.Counter(), time_delta_minutes=1, subdirectory="counter"
        )

    def coordinator(self, counter: counting.Counter) -> Any:
        return lp_utils.StepsLimiter(counter, self._max_executor_steps)

    def trainer(
        self,
        replay: reverb.Client,
        counter: counting.Counter,
    ) -> mava.core.Trainer:
        """The Trainer part of the system."""

        # Create the networks to optimize (online)
        networks = self._network_factory(self._environment_spec)

        # Create system architecture with target networks.
        system_networks = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            policy_networks=networks["policies"],
            shared_weights=self._shared_weights,
        ).create_system()

        dataset = self._builder.make_dataset_iterator(replay)
        counter = counting.Counter(counter, "trainer")
        logger = loggers.make_default_logger("trainer", time_delta=self._log_every)

        return self._builder.make_trainer(
            networks=system_networks,
            dataset=dataset,
            counter=counter,
            logger=logger,
        )

    def executor(
        self,
        replay: reverb.Client,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
    ) -> mava.ParallelEnvironmentLoop:
        """The executor process."""

        # Create the behavior policy.
        networks = self._network_factory(self._environment_spec)

        # Create system architecture with target networks.
        executor_networks = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            policy_networks=networks["policies"],
            shared_weights=self._shared_weights,
        ).create_system()

        # Create the executor.
        executor = self._builder.make_executor(
            policy_networks=executor_networks["behaviors"],
            adder=self._builder.make_adder(replay),
            variable_source=variable_source,
        )

        # Create the environment.
        environment = self._environment_factory(False)

        # Create logger and counter; actors will not spam bigtable.
        counter = counting.Counter(counter, "executor")
        logger = loggers.make_default_logger(
            "executor",
            save_data=False,
            time_delta=self._log_every,
        )

        # Create the loop to connect environment and executor.
        return mava.ParallelEnvironmentLoop(environment, executor, counter, logger)

    def evaluator(
        self,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
        logger: loggers.Logger = None,
    ) -> Any:
        """The evaluation process."""

        # Create the behavior policy.
        networks = self._network_factory(self._environment_spec)
        # networks.init(self._environment_spec)
        # policy_network = networks.make_policy(self._environment_spec)

        # Create system architecture with target networks.
        executor_networks = self._architecture(
            environment_spec=self._environment_spec,
            value_networks=networks["q_networks"],
            policy_networks=networks["policies"],
            shared_weights=self._shared_weights,
        ).create_system()

        # Create the agent.
        executor = self._builder.make_executor(
            policy_networks=executor_networks["policies"],
            variable_source=variable_source,
        )

        # Make the environment.
        environment = self._environment_factory(True)

        # Create logger and counter.
        counter = counting.Counter(counter, "evaluator")
        logger = logger or loggers.make_default_logger(
            "evaluator",
            time_delta=self._log_every,
        )

        # Create the run loop and return it.
        return mava.ParallelEnvironmentLoop(environment, executor, counter, logger)

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
                program.add_node(lp.CourierNode(self.executor, replay, source, counter))

        return program
