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

# type: ignore

# TODO (Kevin): finish DIAL system

"""DIAL system implementation."""
import dataclasses
from typing import Dict, Iterator, Optional

import reverb
import sonnet as snt
import tensorflow as tf
from acme import datasets
from acme.tf import variable_utils
from acme.utils import counting, loggers

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.components.tf.architectures import CentralisedActor
from mava.components.tf.modules.communication import BroadcastedCommunication
from mava.systems import system
from mava.systems.builders import SystemBuilder
from mava.systems.tf.dial.execution import DIALExecutor
from mava.systems.tf.dial.training import DIALTrainer


@dataclasses.dataclass
class DIALConfig:
    """Configuration options for the DIAL system
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
        learning_rate: learning rate for the q-network update.
        discount: discount to use for TD updates.
        logger: logger object to be used by learner.
        checkpoint: boolean indicating whether to checkpoint the learner.
        checkpoint_subpath: directory for the checkpoint.
        policy_networks: if given, this will be used as the policy network.
            Otherwise, an epsilon greedy policy using the online Q network will be
            created. Policy network is used in the actor to sample actions.
        max_gradient_norm: used for gradient clipping.
        replay_table_name: string indicating what name to give the replay table.
    """

    environment_spec: specs.MAEnvironmentSpec
    networks: Dict[str, snt.Module]
    batch_size: int = 256
    prefetch_size: int = 4
    target_update_period: int = 100
    samples_per_insert: float = 32.0
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    importance_sampling_exponent: float = 0.2
    priority_exponent: float = 0.6
    n_step: int = 5
    epsilon: Optional[tf.Tensor] = None
    learning_rate: float = 1e-3
    discount: float = 0.99
    logger: loggers.Logger = None
    checkpoint: bool = True
    checkpoint_subpath: str = "~/mava/"
    policy_networks: Optional[Dict[str, snt.Module]] = None
    max_gradient_norm: Optional[float] = None
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE


class DIALBuilder(SystemBuilder):
    """Builder for DIAL which constructs individual components of the system."""

    """Defines an interface for defining the components of an MARL system.
      Implementations of this interface contain a complete specification of a
      concrete MARL system. An instance of this class can be used to build an
      MARL system which interacts with the environment either locally or in a
      distributed setup.
      """

    def __init__(self, config: DIALConfig):
        """Args:
        config: Configuration options for the DIAL system."""

        self._config = config

        """ _agents: a list of the agent specs (ids).
            _agent_types: a list of the types of agents to be used."""
        self._agents = self._config.environment_spec.get_agent_ids()
        self._agent_types = self._config.environment_spec.get_agent_types()

    def make_replay_table(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> reverb.Table:
        """Create tables to insert data into."""
        return reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Prioritized(self._config.priority_exponent),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=reverb_adders.ParallelNStepTransitionAdder.signature(
                environment_spec
            ),
        )

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
        return reverb_adders.ParallelNStepTransitionAdder(
            priority_fns={self._config.replay_table_name: lambda x: 1.0},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount,
        )

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

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        # Create the actor which defines how we take actions.
        return DIALExecutor(
            policy_networks=policy_networks,
            shared_weights=shared_weights,
            variable_client=variable_client,
            adder=adder,
        )

    def make_trainer(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: Iterator[reverb.ReplaySample],
        huber_loss_parameter: float = 1.0,
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
        max_gradient_norm = self._config.max_gradient_norm
        learning_rate = self._config.learning_rate
        importance_sampling_exponent = self._config.importance_sampling_exponent

        # Create optimizers.
        policy_optimizer = snt.optimizers.Adam(learning_rate=learning_rate)

        # The learner updates the parameters (and initializes them).
        trainer = DIALTrainer(
            agents=agents,
            agent_types=agent_types,
            networks=networks["networks"],
            target_network=networks["target_networks"],
            shared_weights=shared_weights,
            discount=discount,
            importance_sampling_exponent=importance_sampling_exponent,
            policy_optimizer=policy_optimizer,
            target_update_period=target_update_period,
            dataset=dataset,
            huber_loss_parameter=huber_loss_parameter,
            replay_client=replay_client,
            clipping=clipping,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
            max_gradient_norm=max_gradient_norm,
        )
        return trainer


class DIAL(system.System):
    """DIAL system.
    This implements a single-process DIAL system. This is an actor-critic based
    system that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policies of each agent
    (and as a result the behavior) by sampling uniformly from this buffer.
    """

    def __init__(
        self,
        environment_spec: specs.MAEnvironmentSpec,
        networks: Dict[str, snt.Module],
        shared_weights: bool = False,
        batch_size: int = 256,
        prefetch_size: int = 4,
        target_update_period: int = 100,
        samples_per_insert: float = 32.0,
        min_replay_size: int = 1000,
        max_replay_size: int = 1000000,
        importance_sampling_exponent: float = 0.2,
        priority_exponent: float = 0.6,
        n_step: int = 5,
        epsilon: Optional[tf.Tensor] = None,
        learning_rate: float = 1e-3,
        discount: float = 0.99,
        logger: loggers.Logger = None,
        counter: counting.Counter = None,
        checkpoint: bool = True,
        checkpoint_subpath: str = "~/mava/",
        policy_networks: Optional[Dict[str, snt.Module]] = None,
        max_gradient_norm: Optional[float] = None,
        replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
    ):
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
            learning_rate: learning rate for the q-network update.
            discount: discount to use for TD updates.
            logger: logger object to be used by learner.
            checkpoint: boolean indicating whether to checkpoint the learner.
            checkpoint_subpath: directory for the checkpoint.
            policy_networks: if given, this will be used as the policy network.
                Otherwise, an epsilon greedy policy using the online Q network will be
                created. Policy network is used in the actor to sample actions.
            max_gradient_norm: used for gradient clipping.
            replay_table_name: string indicating what name to give the replay table."""

        builder = DIALBuilder(
            DIALConfig(
                environment_spec=environment_spec,
                networks=networks,
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
                learning_rate=learning_rate,
                discount=discount,
                logger=logger,
                counter=counter,
                checkpoint=checkpoint,
                checkpoint_subpath=checkpoint_subpath,
                policy_networks=policy_networks,
                max_gradient_norm=max_gradient_norm,
                replay_table_name=replay_table_name,
            )
        )

        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        replay_table = builder.make_replay_table(environment_spec=environment_spec)
        self._server = reverb.Server([replay_table], port=None)
        replay_client = reverb.Client(f"localhost:{self._server.port}")

        # The adder is used to insert observations into replay.
        adder = builder.make_adder(replay_client)

        # The dataset provides an interface to sample from replay.
        dataset = builder.make_dataset_iterator(replay_client)

        # Create system architecture
        # TODO (Kevin): create decentralised/centralised/networked actor architectures
        # see mava/components/tf/architectures
        architecture = CentralisedActor(
            environment_spec=environment_spec,
            networks=networks,
            shared_weights=shared_weights,
        )

        # Add differentiable communication and get networks
        # TODO (Kevin): create differentiable communication module
        # See mava/components/tf/modules/communication
        networks = BroadcastedCommunication(
            architecture=architecture,
        ).create_system()

        # Create the actor which defines how we take actions.
        executor = builder.make_executor(networks["policies"], adder)

        # The learner updates the parameters (and initializes them).
        trainer = builder.make_trainer(networks, dataset, counter, logger, checkpoint)

        super().__init__(
            executor=executor,
            trainer=trainer,
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=float(batch_size) / samples_per_insert,
        )
