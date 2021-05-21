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

"""MASAC system implementation."""
import copy
from typing import Any, Callable, Dict, Tuple, Type, Union

import acme
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
from acme import specs as acme_specs
from acme.tf import savers as tf2_savers
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava
from mava import core
from mava import specs as mava_specs
from mava.components.tf.architectures import CentralisedSoftQValueActorCritic
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems.tf import executors
from mava.systems.tf.masac import builder, training
from mava.utils import lp_utils
from mava.utils.loggers import MavaLogger
from mava.wrappers import DetailedPerAgentStatistics


class MASAC:
    """MASAC system.
    This implements a single-process SAC system. This is an actor-critic based
    system that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policies of each agent
    (and as a result the behavior) by sampling uniformly from this buffer.
    """

    def __init__(
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        log_info: Tuple,
        architecture: Type[
            CentralisedSoftQValueActorCritic
        ] = CentralisedSoftQValueActorCritic,
        trainer_fn: Union[
            Type[training.BaseMASACTrainer],
        ] = training.CentralisedMASACTrainer,
        executor_fn: Type[core.Executor] = executors.FeedForwardExecutor,
        num_executors: int = 1,
        num_caches: int = 0,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        discount: float = 0.99,
        tau: float = 5e-03,
        temperature: float = 0.9,
        policy_update_frequency: int = 2,
        soft_target_update: bool = False,
        executor_variable_update_period: int = 1000,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_update_period: int = 100,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: float = 32.0,
        policy_optimizer: snt.Optimizer = snt.optimizers.Adam(learning_rate=1e-4),
        critic_V_optimizer: snt.Optimizer = snt.optimizers.Adam(learning_rate=1e-4),
        critic_Q_1_optimizer: snt.Optimizer = snt.optimizers.Adam(learning_rate=1e-4),
        critic_Q_2_optimizer: snt.Optimizer = snt.optimizers.Adam(learning_rate=1e-4),
        n_step: int = 5,
        sequence_length: int = 20,
        period: int = 20,
        sigma: float = 0.3,
        clipping: bool = True,
        log_every: float = 10.0,
        max_executor_steps: int = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        trainer_logger: MavaLogger = None,
        exec_logger: MavaLogger = None,
        eval_logger: MavaLogger = None,
        train_loop_fn: Callable = ParallelEnvironmentLoop,
        eval_loop_fn: Callable = ParallelEnvironmentLoop,
        train_loop_fn_kwargs: Dict = {},
        eval_loop_fn_kwargs: Dict = {},
    ):
        """Initialize the system.
        Args:
            environment_spec: description of the actions, observations, etc.
            policy_networks: the online (optimized) policies for each agent in
                the system.
            critic_V_network: the online critic for predicting state values.
            critic_Q_1_network: the online critic for predicting state-action values.
            critic_Q_2_network: the online critic for predicting state-action values.
            observation_networks: dictionary of optional networks to transform
                the observations before they are fed into any network.
            discount: discount to use for TD updates.
            tau: parameter for soft target update
            temperature: parameter for controlling SAC
            policy_update_frequency: controls the frequency of updating the policy
            soft_target_update: determines wether to use soft or hard target update
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
            clipping: whether to clip gradients by global norm.
            logger: logger object to be used by trainers.
            counter: counter object used to keep track of steps.
            checkpoint: boolean indicating whether to checkpoint the trainers.
            checkpoint_subpath: directory for checkpoints.
            replay_table_name: string indicating what name to give the replay table.
            trainer_logger: logger for trainer class.
            exec_logger: logger for executor.
            eval_logger: logger for evaluator.
            train_loop_fn: loop for training.
            eval_loop_fn: loop for evaluation."""

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
        self._train_loop_fn = train_loop_fn
        self._train_loop_fn_kwargs = train_loop_fn_kwargs
        self._eval_loop_fn = eval_loop_fn
        self._eval_loop_fn_kwargs = eval_loop_fn_kwargs

        if executor_fn == executors.RecurrentExecutor:
            extra_specs = self._get_extra_specs()
        else:
            extra_specs = {}

        self._builder = builder.MASACBuilder(
            builder.MASACConfig(
                environment_spec=environment_spec,
                shared_weights=shared_weights,
                discount=discount,
                tau=tau,
                temperature=temperature,
                policy_update_frequency=policy_update_frequency,
                soft_target_update=soft_target_update,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                executor_variable_update_period=executor_variable_update_period,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                samples_per_insert=samples_per_insert,
                n_step=n_step,
                sequence_length=sequence_length,
                sigma=sigma,
                clipping=clipping,
                checkpoint_subpath=checkpoint_subpath,
            ),
            trainer_fn=trainer_fn,
            executor_fn=executor_fn,
            extra_specs=extra_specs,
        )

    def _get_extra_specs(self) -> Any:
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
        extras = {"core_states": core_state_specs}
        return extras

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
            observation_networks=networks["observations"],
            policy_networks=networks["policies"],
            critic_V_networks=networks["critics_V"],
            critic_Q_1_networks=networks["critics_Q_1"],
            critic_Q_2_networks=networks["critics_Q_2"],
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
            observation_networks=networks["observations"],
            policy_networks=networks["policies"],
            critic_V_networks=networks["critics_V"],
            critic_Q_1_networks=networks["critics_Q_1"],
            critic_Q_2_networks=networks["critics_Q_2"],
            shared_weights=self._shared_weights,
        ).create_system()

        # Create the executor.
        executor = self._builder.make_executor(
            policy_networks=executor_networks["policies"],
            adder=self._builder.make_adder(replay),
            variable_source=variable_source,
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
        """The evaluation process."""

        # Create the behavior policy.
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec
        )

        # Create system architecture with target networks.
        executor_networks = self._architecture(
            environment_spec=self._environment_spec,
            observation_networks=networks["observations"],
            policy_networks=networks["policies"],
            critic_V_networks=networks["critics_V"],
            critic_Q_1_networks=networks["critics_Q_1"],
            critic_Q_2_networks=networks["critics_Q_2"],
            shared_weights=self._shared_weights,
        ).create_system()

        # Create the agent.
        executor = self._builder.make_executor(
            policy_networks=executor_networks["policies"],
            variable_source=variable_source,
        )

        # Make the environment.
        environment = self._environment_factory(evaluation=True)  # type: ignore

        # Create logger and counter.
        counter = counting.Counter(counter, "evaluator")

        # Create the run loop and return it.
        # Create the loop to connect environment and executor.
        eval_loop = ParallelEnvironmentLoop(
            environment,
            executor,
            counter=counter,
            logger=self._eval_logger,
            **self._eval_loop_fn_kwargs,
        )

        eval_loop = DetailedPerAgentStatistics(eval_loop)
        return eval_loop

    def build(self, name: str = "masac") -> Any:
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
