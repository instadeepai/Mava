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

"""MAPPO system implementation."""
from typing import Any, Callable, Dict, Optional, Tuple, Type

import acme
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
from acme import specs as acme_specs
from acme.utils import counting, loggers

import mava
from mava import core
from mava import specs as mava_specs
from mava.components.tf.architectures import DecentralisedValueActorCritic
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import savers as tf2_savers
from mava.systems.tf.mappo import builder, execution, training
from mava.utils import lp_utils
from mava.utils.loggers import Logger
from mava.wrappers import DetailedPerAgentStatistics


class MAPPO:

    """MAPPO system.
    This implements a single-process MAPPO system. This is an actor-critic based
    system that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policies of each agent.
    """

    def __init__(
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        log_info: Tuple,
        architecture: Type[
            DecentralisedValueActorCritic
        ] = DecentralisedValueActorCritic,
        trainer_fn: Type[training.MAPPOTrainer] = training.MAPPOTrainer,
        executor_fn: Type[core.Executor] = execution.MAPPOFeedForwardExecutor,
        num_executors: int = 1,
        num_caches: int = 0,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        executor_variable_update_period: int = 100,
        policy_optimizer: snt.Optimizer = snt.optimizers.Adam(learning_rate=5e-4),
        critic_optimizer: snt.Optimizer = snt.optimizers.Adam(learning_rate=1e-5),
        discount: float = 0.99,
        lambda_gae: float = 0.99,
        clipping_epsilon: float = 0.2,
        entropy_cost: float = 0.01,
        baseline_cost: float = 0.5,
        max_abs_reward: Optional[float] = None,
        max_gradient_norm: Optional[float] = None,
        max_queue_size: int = 100000,
        batch_size: int = 16,
        sequence_length: int = 10,
        sequence_period: int = 5,
        log_every: float = 10.0,
        max_executor_steps: int = None,
        checkpoint: bool = True,
    ):

        """Initialize the system.
        Args:
        environment_spec: description of the actions, observations, etc.
        networks: ...
        sequence_length: ...
        sequence_period: ...
        shared_weights: ...
        counter: ...
        entropy_cost: ...
        baseline_cost: ...
        clipping_epsilon: ...
        max_abs_reward: ...
        batch_size: batch size for updates.
        max_queue_size: maximum queue size.
        discount: discount to use for TD updates.
        logger: logger object to be used by learner.
        max_gradient_norm: used for gradient clipping.
        checkpoint: ...
        replay_table_name: string indicating what name to give the replay table."""

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

        self._builder = builder.MAPPOBuilder(
            config=builder.MAPPOConfig(
                environment_spec=environment_spec,
                shared_weights=shared_weights,
                executor_variable_update_period=executor_variable_update_period,
                discount=discount,
                lambda_gae=lambda_gae,
                clipping_epsilon=clipping_epsilon,
                entropy_cost=entropy_cost,
                baseline_cost=baseline_cost,
                max_abs_reward=max_abs_reward,
                max_gradient_norm=max_gradient_norm,
                max_queue_size=max_queue_size,
                batch_size=batch_size,
                sequence_length=sequence_length,
                sequence_period=sequence_period,
                checkpoint=checkpoint,
                policy_optimizer=policy_optimizer,
                critic_optimizer=critic_optimizer,
            ),
        )

    def replay(self) -> Any:
        """The replay storage."""
        return self._builder.make_replay_tables(self._environment_spec)

    def counter(self) -> Any:
        return tf2_savers.CheckpointingRunner(
            counting.Counter(), time_delta_minutes=1, subdirectory="counter"
        )

    def coordinator(self, counter: counting.Counter) -> Any:
        return lp_utils.StepsLimiter(counter, self._max_executor_steps)  # type: ignore

    def trainer(
        self,
        replay: reverb.Client,
        counter: counting.Counter,
    ) -> mava.core.Trainer:
        """The Trainer part of the system."""

        # get log info
        log_dir, log_time_stamp = self._log_info

        # Create the networks to optimize (online)
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec
        )

        # Create system architecture with target networks.
        system_networks = self._architecture(
            environment_spec=self._environment_spec,
            observation_networks=networks["observations"],
            policy_networks=networks["policies"],
            critic_networks=networks["critics"],
            shared_weights=self._shared_weights,
        ).create_system()

        dataset = self._builder.make_dataset_iterator(replay)
        counter = counting.Counter(counter, "trainer")
        trainer_logger = Logger(
            label="system_trainer",
            directory=log_dir,
            to_terminal=True,
            to_tensorboard=True,
            time_stamp=log_time_stamp,
            time_delta=self._log_every,
        )

        return self._builder.make_trainer(
            networks=system_networks,
            dataset=dataset,
            counter=counter,
            logger=trainer_logger,
        )

    def executor(
        self,
        executor_id: str,
        replay: reverb.Client,
        variable_source: acme.VariableSource,
        counter: counting.Counter,
    ) -> mava.ParallelEnvironmentLoop:
        """The executor process."""

        # get log info
        log_dir, log_time_stamp = self._log_info

        # Create the behavior policy.
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec
        )

        # Create system architecture with target networks.
        system = self._architecture(
            environment_spec=self._environment_spec,
            observation_networks=networks["observations"],
            policy_networks=networks["policies"],
            critic_networks=networks["critics"],
            shared_weights=self._shared_weights,
        )

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
        train_logger = Logger(
            label=f"train_loop_executor_{executor_id}",
            directory=log_dir,
            to_terminal=True,
            to_tensorboard=True,
            time_stamp=log_time_stamp,
            time_delta=self._log_every,
        )

        # Create the loop to connect environment and executor.
        train_loop = ParallelEnvironmentLoop(
            environment, executor, counter=counter, logger=train_logger
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

        # get log info
        log_dir, log_time_stamp = self._log_info

        # Create the behavior policy.
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec
        )

        # Create system architecture with target networks.
        system = self._architecture(
            environment_spec=self._environment_spec,
            observation_networks=networks["observations"],
            policy_networks=networks["policies"],
            critic_networks=networks["critics"],
            shared_weights=self._shared_weights,
        )

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
        eval_logger = Logger(
            label="eval_loop",
            directory=log_dir,
            to_terminal=True,
            to_tensorboard=True,
            time_stamp=log_time_stamp,
            time_delta=self._log_every,
        )

        # Create the run loop and return it.
        # Create the loop to connect environment and executor.
        eval_loop = ParallelEnvironmentLoop(
            environment, executor, counter=counter, logger=eval_logger
        )

        eval_loop = DetailedPerAgentStatistics(eval_loop)
        return eval_loop

    def build(self, name: str = "mappo") -> Any:
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
