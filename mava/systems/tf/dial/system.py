# Copyright 2021 [...placeholder...]. All rights reserved.
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
import copy
from typing import Any, Callable, Dict, Optional, Tuple, Type

import acme
import dm_env
import launchpad as lp
import reverb
import sonnet as snt
import tensorflow as tf
from acme import specs as acme_specs
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

import mava
from mava import core
from mava import specs as mava_specs
from mava.adders import reverb as reverb_adders
from mava.components.tf.architectures import DecentralisedPolicyActor
from mava.components.tf.modules.communication import BroadcastedCommunication
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems import system
from mava.systems.tf import savers as tf2_savers
from mava.systems.tf.dial import builder
from mava.systems.tf.dial.execution import DIALExecutor
from mava.utils import lp_utils
from mava.utils.loggers import MavaLogger
from mava.wrappers import DetailedPerAgentStatistics


class DIAL(system.System):
    """DIAL system.
    This implements a single-process DIAL system. This is an actor-critic based
    system that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policies of each agent
    (and as a result the behavior) by sampling uniformly from this buffer.
    """

    def __init__(
        self,
        environment_factory: Callable[[bool], dm_env.Environment],
        network_factory: Callable[[acme_specs.BoundedArray], Dict[str, snt.Module]],
        architecture: Type[DecentralisedPolicyActor] = DecentralisedPolicyActor,
        executor_fn: Type[core.Executor] = DIALExecutor,
        log_info: Tuple = None,
        num_executors: int = 1,
        num_caches: int = 0,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        discount: float = 1.0,
        batch_size: int = 32,
        prefetch_size: int = 4,
        target_update_period: int = 100,
        min_replay_size: int = 100,
        max_replay_size: int = 1000000,
        samples_per_insert: float = 32.0,
        policy_optimizer: snt.Optimizer = snt.optimizers.RMSProp(5e-4, momentum=0.95),
        n_step: int = 1,
        sequence_length: int = 20,
        period: int = 20,
        importance_sampling_exponent: float = 0.2,
        priority_exponent: float = 0.6,
        epsilon: Optional[tf.Tensor] = None,
        counter: counting.Counter = None,
        max_executor_steps: int = None,
        checkpoint: bool = False,
        checkpoint_subpath: str = "~/mava/",
        max_gradient_norm: Optional[float] = None,
        replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
        communication_module: Type[BroadcastedCommunication] = BroadcastedCommunication,
        trainer_logger: MavaLogger = None,
        exec_logger: MavaLogger = None,
        eval_logger: MavaLogger = None,
    ):
        # TODO: Update args
        """Initialize the system.
        Args:
            environment_spec: description of the actions, observations, etc.
            networks: the online Q network (the one being optimized)
            batch_size: batch size for updates.
            prefetch_size: size to prefetch from replay.
            target_update_period: number of learner steps to perform before updating
                the target networks.
            samples_per_insert: number of samples to take from replay for every insert
                that is made.
            min_replay_size: minimum replay size before updating. This and all
                following arguments are related to dataset construction and will be
                ignored if a dataset argument is passed.
            max_replay_size: maximum replay size.
            importance_sampling_exponent: power to which importance weights are raised
                before normalizing.
            priority_exponent: exponent used in prioritized sampling.
            n_step: number of steps to squash into a single transition.
            epsilon: probability of taking a random action; ignored if a policy
                network is given.
            discount: discount to use for TD updates.
            logger: logger object to be used by learner.
            checkpoint: boolean indicating whether to checkpoint the learner.
            checkpoint_subpath: directory for the checkpoint.
            policy_networks: if given, this will be used as the policy network.
                Otherwise, an epsilon greedy policy using the online Q network will be
                created. Policy network is used in the actor to sample actions.
            max_gradient_norm: used for gradient clipping.
            replay_table_name: string indicating what name to give the replay table."""

        if not environment_spec:
            environment_spec = mava_specs.MAEnvironmentSpec(
                environment_factory(evaluation=False)  # type: ignore
            )

        self._architecture_fn = architecture
        self._communication_module_fn = communication_module
        self._environment_factory = environment_factory
        self._network_factory = network_factory
        self._log_info = log_info
        self._environment_spec = environment_spec
        self._shared_weights = shared_weights
        self._num_exectors = num_executors
        self._num_caches = num_caches
        self._max_executor_steps = max_executor_steps
        self._checkpoint_subpath = checkpoint_subpath
        self._checkpoint = checkpoint
        self._trainer_logger = trainer_logger
        self._exec_logger = exec_logger
        self._eval_logger = eval_logger

        extra_specs = self._get_extra_specs()

        self._builder = builder.DIALBuilder(
            builder.DIALConfig(
                environment_spec=environment_spec,
                shared_weights=shared_weights,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                samples_per_insert=samples_per_insert,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                importance_sampling_exponent=importance_sampling_exponent,
                priority_exponent=priority_exponent,
                n_step=n_step,
                epsilon=epsilon,
                discount=discount,
                counter=counter,
                checkpoint=checkpoint,
                checkpoint_subpath=checkpoint_subpath,
                max_gradient_norm=max_gradient_norm,
                replay_table_name=replay_table_name,
                policy_optimizer=policy_optimizer,
                sequence_length=sequence_length,
                period=period,
            ),
            executor_fn=executor_fn,
            extra_specs=extra_specs,
        )

        # TODO (Kevin): create decentralised/centralised/networked actor architectures
        # see mava/components/tf/architectures

        # TODO (Kevin): create differentiable communication module
        # See mava/components/tf/modules/communication

    def _get_extra_specs(self) -> Any:
        agents = self._environment_spec.get_agent_ids()
        networks = self._network_factory(  # type: ignore
            environment_spec=self._environment_spec
        )
        core_state_spec = {}
        for agent in agents:
            agent_type = agent.split("_")[0]
            core_state_spec[agent] = {
                "state": tf2_utils.squeeze_batch_dim(
                    networks["policies"][agent_type].initial_state(1)
                ),
                "message": tf2_utils.squeeze_batch_dim(
                    networks["policies"][agent_type].initial_message(1)
                ),
            }
        extras = {"core_states": core_state_spec}
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
        architecture = self._architecture_fn(
            environment_spec=self._environment_spec,
            observation_networks=networks["observations"],
            policy_networks=networks["policies"],
            shared_weights=self._shared_weights,
        )

        communication_module = self._communication_module_fn(
            architecture=architecture,
            shared=True,
            channel_size=1,
            channel_noise=0,
        )

        system_networks = communication_module.create_system()

        dataset = self._builder.make_dataset_iterator(replay)
        counter = counting.Counter(counter, "trainer")

        system_networks["communication_module"] = {"all_agents": communication_module}

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
        architecture = self._architecture_fn(
            environment_spec=self._environment_spec,
            observation_networks=networks["observations"],
            policy_networks=networks["policies"],
            shared_weights=self._shared_weights,
        )

        communication_module = self._communication_module_fn(
            architecture=architecture,
            shared=True,
            channel_size=1,
            channel_noise=0,
        )

        _ = communication_module.create_system()

        # behaviour policy networks (obs net + policy head)
        behaviour_policy_networks = communication_module.create_behaviour_policy()

        # Create the executor.
        executor = self._builder.make_executor(
            policy_networks={
                "policy_net": behaviour_policy_networks,
                "communication_module": communication_module,
            },
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
        architecture = self._architecture_fn(
            environment_spec=self._environment_spec,
            observation_networks=networks["observations"],
            policy_networks=networks["policies"],
            shared_weights=self._shared_weights,
        )

        communication_module = self._communication_module_fn(
            architecture=architecture,
            shared=True,
            channel_size=1,
            channel_noise=0,
        )

        _ = communication_module.create_system()

        # behaviour policy networks (obs net + policy head)
        behaviour_policy_networks = communication_module.create_behaviour_policy()

        # Create the executor.
        executor = self._builder.make_executor(
            policy_networks={
                "policy_net": behaviour_policy_networks,
                "communication_module": communication_module,
            },
            variable_source=variable_source,
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

    def coordinator(self, counter: counting.Counter) -> Any:
        return lp_utils.StepsLimiter(counter, self._max_executor_steps)  # type: ignore

    def build(self, name: str = "maddpg") -> Any:
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
