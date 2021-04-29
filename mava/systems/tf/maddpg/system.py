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
import dataclasses
from typing import Dict, Iterator, List, Optional, Type

import reverb
import sonnet as snt
from acme import datasets
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils
from acme.utils import counting, loggers

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.components.tf.architectures import DecentralisedQValueActorCritic
from mava.systems import system
from mava.systems.builders import SystemBuilder
from mava.systems.tf import executors
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


class MADDPG(system.System):
    """MADDPG system.
    This implements a single-process DDPG system. This is an actor-critic based
    system that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policies of each agent
    (and as a result the behavior) by sampling uniformly from this buffer.
    """

    def __init__(
        self,
        environment_spec: specs.MAEnvironmentSpec,
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        observation_networks: Dict[str, snt.Module],
        trainer_fn: Type[
            training.BaseMADDPGTrainer
        ] = training.DecentralisedMADDPGTrainer,
        executer_fn: Type[core.Executor] = executors.FeedForwardExecutor,
        shared_weights: bool = True,
        discount: float = 0.99,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_update_period: int = 100,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        samples_per_insert: float = 32.0,
        n_step: int = 5,
        sigma: float = 0.3,
        clipping: bool = True,
        logger: loggers.Logger = None,
        counter: counting.Counter = None,
        checkpoint: bool = True,
        replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
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

        builder = MADDPGBuilder(
            MADDPGConfig(
                environment_spec=environment_spec,
                policy_networks=policy_networks,
                critic_networks=critic_networks,
                observation_networks=observation_networks,
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
                logger=logger,
                counter=counter,
                checkpoint=checkpoint,
                replay_table_name=replay_table_name,
            ),
            trainer_fn=trainer_fn,
            executer_fn=executer_fn,
        )

        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        replay_tables = builder.make_replay_tables(environment_spec=environment_spec)
        self._server = reverb.Server(replay_tables, port=None)
        replay_client = reverb.Client(f"localhost:{self._server.port}")

        # The adder is used to insert observations into replay.
        adder = builder.make_adder(replay_client)

        # The dataset provides an interface to sample from replay.
        dataset = builder.make_dataset_iterator(replay_client)

        # Create the networks
        networks = DecentralisedQValueActorCritic(
            environment_spec=environment_spec,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            shared_weights=shared_weights,
        ).create_system()

        # Create the actor which defines how we take actions.
        executor = builder.make_executor(networks["policies"], adder)

        # The learner updates the parameters (and initializes them).
        trainer = builder.make_trainer(
            networks=networks,
            dataset=dataset,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
        )

        # NB If using both NetworkStatistics and TrainerStatistics, order is important.
        # NetworkStatistics needs to appear before TrainerStatistics.
        trainer = NetworkStatisticsActorCritic(trainer)

        trainer = DetailedTrainerStatistics(  # type: ignore
            trainer, metrics=["policy_loss", "critic_loss"]
        )

        super().__init__(
            executor=executor,
            trainer=trainer,
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=float(batch_size) / samples_per_insert,
        )
