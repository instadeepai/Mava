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
from typing import Any, Callable, Dict, Tuple, Type

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
from mava.components.tf.architectures import DecentralisedQValueActorCritic
from mava.environment_loop import ParallelEnvironmentLoop
from mava.systems import system
from mava.systems.tf import executors
<<<<<<< HEAD
from mava.systems.tf.maddpg import builder, training
from mava.utils import lp_utils
from mava.utils.loggers import Logger
from mava.wrappers import DetailedPerAgentStatistics
=======
from mava.systems.tf.maddpg import training
from mava.wrappers import DetailedTrainerStatistics, NetworkStatisticsActorCritic


@dataclasses.dataclass
class MADDPGConfig:
    """Configuration options for the MADDPG system.
    Args:
            environment_spec: description of the actions, observations, etc.
            policy_networks: the online (optimized) policies for each agent in
                the system.
            critic_networks: the online critic for each agent in the system.
            observation_networks: dictionary of optional networks to transform
                the observations before they are fed into any network.
            discount: discount to use for TD updates.
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
            replay_table_name: string indicating what name to give the replay table."""

    environment_spec: specs.MAEnvironmentSpec
    policy_networks: Dict[str, snt.Module]
    critic_networks: Dict[str, snt.Module]
    observation_networks: Dict[str, snt.Module]
    shared_weights: bool = True
    discount: float = 0.99
    batch_size: int = 256
    prefetch_size: int = 4
    target_update_period: int = 100
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    samples_per_insert: float = 32.0
    n_step: int = 5
    sequence_length: int = 20
    period: int = 20
    sigma: float = 0.3
    clipping: bool = True
    logger: loggers.Logger = None
    counter: counting.Counter = None
    checkpoint: bool = True
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE


class MADDPGBuilder(SystemBuilder):
    """Builder for MADDPG which constructs individual components of the system."""

    """Defines an interface for defining the components of an RL system.
      Implementations of this interface contain a complete specification of a
      concrete RL system. An instance of this class can be used to build an
      RL system which interacts with the environment either locally or in a
      distributed setup.
      """

    def __init__(
        self,
        config: MADDPGConfig,
        trainer_fn: Type[
            training.BaseMADDPGTrainer
        ] = training.DecentralisedMADDPGTrainer,
        executer_fn: Type[core.Executor] = executors.FeedForwardExecutor,
    ):
        """Args:
        config: Configuration options for the MADDPG system.
        trainer_fn: Trainer module to use."""

        self._config = config

        """ _agents: a list of the agent specs (ids).
            _agent_types: a list of the types of agents to be used."""
        self._agents = self._config.environment_spec.get_agent_ids()
        self._agent_types = self._config.environment_spec.get_agent_types()
        self._trainer_fn = trainer_fn
        self._executer_fn = executer_fn

    def make_replay_tables(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""

        # Select adder
        if self._executer_fn == executors.FeedForwardExecutor:
            adder = reverb_adders.ParallelNStepTransitionAdder.signature(
                environment_spec
            )
        elif self._executer_fn == executors.RecurrentExecutor:
            core_state_spec = {}
            for agent in self._agents:
                agent_type = agent.split("_")[0]
                core_state_spec[agent] = (
                    tf2_utils.squeeze_batch_dim(
                        self._config.policy_networks[agent_type].initial_state(1)
                    ),
                )
            adder = reverb_adders.ParallelSequenceAdder.signature(
                environment_spec, core_state_spec
            )
        else:
            raise NotImplementedError("Unknown executor type: ", self._executer_fn)

        return [
            reverb.Table(
                name=self._config.replay_table_name,
                sampler=reverb.selectors.Uniform(),
                remover=reverb.selectors.Fifo(),
                max_size=self._config.max_replay_size,
                rate_limiter=reverb.rate_limiters.MinSize(1),
                signature=adder,
            )
        ]

    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the system."""
        dataset = datasets.make_reverb_dataset(
            table=self._config.replay_table_name,
            server_address=replay_client.server_address,
            batch_size=self._config.batch_size,
            prefetch_size=self._config.prefetch_size,
        )
        return iter(dataset)

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> Optional[adders.ParallelAdder]:
        """Create an adder which records data generated by the executor/environment.
        Args:
          replay_client: Reverb Client which points to the replay server.
        """

        # Select adder
        if self._executer_fn == executors.FeedForwardExecutor:
            adder = reverb_adders.ParallelNStepTransitionAdder(
                priority_fns=None,  # {self._config.replay_table_name: lambda x: 1.0},
                client=replay_client,
                n_step=self._config.n_step,
                discount=self._config.discount,
            )
        elif self._executer_fn == executors.RecurrentExecutor:
            adder = reverb_adders.ParallelSequenceAdder(
                priority_fns=None,  # {self._config.replay_table_name: lambda x: 1.0},
                client=replay_client,
                sequence_length=self._config.sequence_length,
                period=self._config.period,
            )
        else:
            raise NotImplementedError("Unknown executor type: ", self._executer_fn)

        return adder

    def make_executor(
        self,
        policy_networks: Dict[str, snt.Module],
        adder: Optional[adders.ParallelAdder] = None,
        variable_source: Optional[core.VariableSource] = None,
    ) -> core.Executor:
        """Create an executor instance.
        Args:
          policy_networks: A struct of instance of all the different policy networks;
           this should be a callable
            which takes as input observations and returns actions.
          adder: How data is recorded (e.g. added to replay).
          variable_source: A source providing the necessary executor parameters.
        """
        shared_weights = self._config.shared_weights

        variable_client = None
        if variable_source:
            agent_keys = self._agent_types if shared_weights else self._agents

            # Create policy variables
            variables = {}
            for agent in agent_keys:
                variables[agent] = policy_networks[agent].variables

            # Get new policy variables
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={"policy": variables},
                update_period=1000,
            )

            # Update variables
            # TODO: Is this needed? Probably not because
            #  in acme they only update policy.variables.
            # for agent in agent_keys:
            #     policy_networks[agent].variables = variables[agent]

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        # Create the actor which defines how we take actions.
        return self._executer_fn(
            policy_networks=policy_networks,
            shared_weights=shared_weights,
            variable_client=variable_client,
            adder=adder,
        )

    def make_trainer(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: Iterator[reverb.ReplaySample],
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[types.NestedLogger] = None,
        checkpoint: bool = False,
    ) -> core.Trainer:
        """Creates an instance of the trainer.
        Args:
          networks: struct describing the networks needed by the trainer; this can
            be specific to the trainer in question.
          dataset: iterator over samples from replay.
          replay_client: client which allows communication with replay, e.g. in
            order to update priorities.
          counter: a Counter which allows for recording of counts (trainer steps,
            executor steps, etc.) distributed throughout the system.
          logger: Logger object for logging metadata.
          checkpoint: bool controlling whether the trainer checkpoints itself.
        """
        agents = self._agents
        agent_types = self._agent_types
        shared_weights = self._config.shared_weights
        clipping = self._config.clipping
        discount = self._config.discount
        target_update_period = self._config.target_update_period

        # Create optimizers.
        policy_optimizer = snt.optimizers.Adam(learning_rate=1e-4)
        critic_optimizer = snt.optimizers.Adam(learning_rate=1e-4)

        # The learner updates the parameters (and initializes them).
        trainer = self._trainer_fn(
            agents=agents,
            agent_types=agent_types,
            policy_networks=networks["policies"],
            critic_networks=networks["critics"],
            observation_networks=networks["observations"],
            target_policy_networks=networks["target_policies"],
            target_critic_networks=networks["target_critics"],
            target_observation_networks=networks["target_observations"],
            shared_weights=shared_weights,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            clipping=clipping,
            discount=discount,
            target_update_period=target_update_period,
            dataset=dataset,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
        )
        return trainer
>>>>>>> d2ab48f5fe095dcc906f14cb40239c10eadeb092


class MADDPG(system.System):
    """MADDPG system.
    This implements a single-process DDPG system. This is an actor-critic based
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
            DecentralisedQValueActorCritic
        ] = DecentralisedQValueActorCritic,
        trainer_fn: Type[
            training.BaseMADDPGTrainer
        ] = training.DecentralisedMADDPGTrainer,
        executer_fn: Type[core.Executor] = executors.FeedForwardExecutor,
        num_executors: int = 1,
        num_caches: int = 0,
        environment_spec: mava_specs.MAEnvironmentSpec = None,
        shared_weights: bool = True,
        discount: float = 0.99,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_update_period: int = 100,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: float = 32.0,
        policy_optimizer: snt.Optimizer = None,
        critic_optimizer: snt.Optimizer = None,
        n_step: int = 5,
        sigma: float = 0.3,
        clipping: bool = True,
        log_every: float = 10.0,
        max_executor_steps: int = None,
    ):
        """Initialize the system.
        Args:
            environment_spec: description of the actions, observations, etc.
            policy_networks: the online (optimized) policies for each agent in
                the system.
            critic_networks: the online critic for each agent in the system.
            observation_networks: dictionary of optional networks to transform
                the observations before they are fed into any network.
            discount: discount to use for TD updates.
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

        self._builder = builder.MADDPGBuilder(
            builder.MADDPGConfig(
                environment_spec=environment_spec,
                shared_weights=shared_weights,
                discount=discount,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                min_replay_size=min_replay_size,
                max_replay_size=max_replay_size,
                samples_per_insert=samples_per_insert,
                n_step=n_step,
                sigma=sigma,
                clipping=clipping,
            ),
            trainer_fn=trainer_fn,
            executer_fn=executer_fn,
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
        executor_networks = self._architecture(
            environment_spec=self._environment_spec,
            observation_networks=networks["observations"],
            policy_networks=networks["policies"],
            critic_networks=networks["critics"],
            shared_weights=self._shared_weights,
        ).create_system()

        # Create the executor.
        # TODO (Arnu): getting "method get_variables not found" error
        # when passing in var source directly from launchpad build function.
        # This is because the var source being sent in seems to be a courier
        # node of type courier.python.client.Client and not a variable source.
        # Need to investigate further.
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
        train_logger = Logger(
            label=f"train_loop_executor_{executor_id}",
            directory=log_dir,
            to_terminal=True,
            to_tensorboard=True,
            time_stamp=log_time_stamp,
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
        executor_networks = self._architecture(
            environment_spec=self._environment_spec,
            observation_networks=networks["observations"],
            policy_networks=networks["policies"],
            critic_networks=networks["critics"],
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
        eval_logger = Logger(
            label="eval_loop",
            directory=log_dir,
            to_terminal=True,
            to_tensorboard=True,
            time_stamp=log_time_stamp,
        )

        # Create the run loop and return it.
        # Create the loop to connect environment and executor.
        eval_loop = ParallelEnvironmentLoop(
            environment, executor, counter=counter, logger=eval_logger
        )

        eval_loop = DetailedPerAgentStatistics(eval_loop)
        return eval_loop

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
