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

# TODO (StJohn): finish Qmix

"""QMIX system implementation."""
import dataclasses
from typing import Dict, Iterator, Optional, Type, Union

import reverb
import sonnet as snt
import tensorflow as tf
from acme import datasets

# from acme.tf import variable_utils
from acme.utils import counting, loggers

from mava import adders, core, specs
from mava.adders import reverb as reverb_adders
from mava.components.tf.architectures import DecentralisedActor
from mava.components.tf.modules.mixing import MonotonicMixing
from mava.systems import system
from mava.systems.builders import SystemBuilder
from mava.systems.tf import executors
from mava.systems.tf.qmix import training

NestedLogger = Union[loggers.Logger, Dict[str, loggers.Logger]]


@dataclasses.dataclass
class QMIXConfig:
    """Configuration options for the QMIX system
    Args:
        environment_spec: description of the actions, observations, etc.
        q_networks: the online Q network (the one being optimized)
        behavior_networks: the network responsible for data collection (learning).
        observation_networks: dictionary of optional networks to transform
                the observations before they are fed into any network.
        epsilon: probability of taking a random action; ignored if a policy
            network is given.
        shared_weights: boolean determining whether shared weights is used.
        target_update_period: number of learner steps to perform before updating
            the target networks.
        clipping: whether to clip gradients by global norm.
        replay_table_name: string indicating what name to give the replay table.
        max_replay_size: maximum replay size.
        samples_per_insert: number of samples to take from replay for every insert
            that is made.
        prefetch_size: size to prefetch from replay.
        batch_size: batch size for updates.
        n_step: number of steps to squash into a single transition.
        discount: discount to use for TD updates.
        counter: counter object used to keep track of steps.
        logger: logger object to be used by trainers.
        checkpoint: boolean indicating whether to checkpoint the learner.
    """

    environment_spec: specs.MAEnvironmentSpec
    q_networks: Dict[str, snt.Module]
    behavior_networks: Dict[str, snt.Module]
    observation_networks: Dict[str, snt.Module]
    epsilon: tf.Variable
    shared_weights: bool
    target_update_period: int
    clipping: bool
    replay_table_name: str
    max_replay_size: int
    samples_per_insert: float
    prefetch_size: int
    batch_size: int
    n_step: int
    discount: float
    counter: counting.Counter
    logger: loggers.Logger
    checkpoint: bool


class QMIXBuilder(SystemBuilder):
    """Builder for QMIX which constructs individual components of the system."""

    """Defines an interface for defining the components of an MARL system.
      Implementations of this interface contain a complete specification of a
      concrete MARL system. An instance of this class can be used to build an
      MARL system which interacts with the environment either locally or in a
      distributed setup.
      """

    def __init__(
        self,
        config: QMIXConfig,
        trainer_fn: Type[training.QMIXTrainer] = training.QMIXTrainer,
    ):
        """Args:
        _config: Configuration options for the MADDPG system.
        _trainer_fn: Trainer module to use."""
        self._config = config
        self._trainer_fn = trainer_fn

    def make_replay_table(
        self,
        environment_spec: specs.EnvironmentSpec,
    ) -> reverb.Table:
        """Create tables to insert data into."""
        return reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(self._config.batch_size),
            signature=reverb_adders.ParallelNStepTransitionAdder.signature(
                environment_spec
            ),
        )

    def make_dataset_iterator(
        self, replay_client: reverb.Client
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
        self, replay_client: reverb.Client
    ) -> Optional[adders.ParallelAdder]:
        """Create an adder which records data generated by the executor/environment.
        Args:
          replay_client: Reverb Client which points to the replay server."""
        return reverb_adders.ParallelNStepTransitionAdder(
            client=replay_client,
            priority_fns=None,
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
            behavior_networks: A struct of instance of all
                the different behaviour networks,
                this should be a callable which takes as input observations
                and returns actions.
            adder: How data is recorded (e.g. added to replay).
        """

        # Create the executor which coordinates the actors.
        return executors.FeedForwardExecutor(
            policy_networks=policy_networks,
            shared_weights=self._config.shared_weights,
            variable_client=None,
            adder=adder,
        )

    def make_trainer(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: Iterator[reverb.ReplaySample],
        replay_client: Optional[reverb.Client] = None,
        counter: Optional[counting.Counter] = None,
        logger: Optional[NestedLogger] = None,
        # TODO: eliminate checkpoint and move it outside.
        checkpoint: bool = False,
    ) -> core.Trainer:
        """Creates an instance of the trainer.
        Args:
          networks: struct describing the networks needed by the trainer; this can
            be specific to the trainer in question.
          dataset: iterator over samples from replay.
          counter: a Counter which allows for recording of counts (trainer steps,
            executor steps, etc.) distributed throughout the system.
          logger: Logger object for logging metadata.
          checkpoint: bool controlling whether the trainer checkpoints itself.
        """
        q_networks = networks["q_networks"]
        target_q_networks = networks["target_q_networks"]
        observation_networks = networks["observation_networks"]
        mixing_networks = networks["mixing_networks"]

        agents = self._config.environment_spec.get_agent_ids()
        agent_types = self._config.environment_spec.get_agent_types()

        # Create optimizers.
        optimizer = snt.optimizers.Adam(learning_rate=1e-3)

        # The learner updates the parameters (and initializes them).
        trainer = self._trainer_fn(
            agents=agents,
            agent_types=agent_types,
            q_networks=q_networks,
            target_q_networks=target_q_networks,
            observation_networks=observation_networks,
            mixing_networks=mixing_networks,
            epsilon=self._config.epsilon,
            shared_weights=self._config.shared_weights,
            optimizer=optimizer,
            target_update_period=self._config.target_update_period,
            clipping=self._config.clipping,
            dataset=dataset,
            counter=self._config.counter,
            logger=self._config.logger,
            checkpoint=self._config.checkpoint,
        )

        return trainer


class QMIX(system.System):
    """QMIX system.
    This implements a single-process QMIX system.
    Args:
        environment_spec: description of the actions, observations, etc.
        q_networks: the online Q network (the one being optimized)
        behavior_networks: the network responsible for data collection (learning).
        observation_networks: dictionary of optional networks to transform
                the observations before they are fed into any network.
        epsilon: probability of taking a random action; ignored if a policy
            network is given.
        trainer_fn: the class used for training the agent and mixing networks.
        shared_weights: boolean determining whether shared weights is used.
        target_update_period: number of learner steps to perform before updating
            the target networks.
        clipping: whether to clip gradients by global norm.
        replay_table_name: string indicating what name to give the replay table.
        max_replay_size: maximum replay size.
        samples_per_insert: number of samples to take from replay for every insert
            that is made.
        prefetch_size: size to prefetch from replay.
        batch_size: batch size for updates.
        n_step: number of steps to squash into a single transition.
        discount: discount to use for TD updates.
        counter: counter object used to keep track of steps.
        logger: logger object to be used by trainers.
        checkpoint: boolean indicating whether to checkpoint the learner.
    """

    def __init__(
        self,
        environment_spec: specs.MAEnvironmentSpec,
        q_networks: Dict[str, snt.Module],
        behavior_networks: Dict[str, snt.Module],
        observation_networks: Dict[str, snt.Module],
        epsilon: tf.Variable,
        trainer_fn: Type[training.QMIXTrainer] = training.QMIXTrainer,
        shared_weights: bool = False,
        target_update_period: int = 100,
        clipping: bool = False,
        replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
        max_replay_size: int = 1_000_000,
        samples_per_insert: float = 32.0,
        prefetch_size: int = 4,
        batch_size: int = 256,
        n_step: int = 5,
        discount: float = 0.99,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = False,
    ):
        """Initialize the system."""

        builder = QMIXBuilder(
            QMIXConfig(
                environment_spec=environment_spec,
                q_networks=q_networks,
                behavior_networks=behavior_networks,
                observation_networks=observation_networks,
                shared_weights=shared_weights,
                discount=discount,
                epsilon=epsilon,
                batch_size=batch_size,
                prefetch_size=prefetch_size,
                target_update_period=target_update_period,
                max_replay_size=max_replay_size,
                samples_per_insert=samples_per_insert,
                n_step=n_step,
                clipping=clipping,
                logger=logger,
                counter=counter,
                checkpoint=checkpoint,
                replay_table_name=replay_table_name,
            ),
            trainer_fn=trainer_fn,
        )

        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        replay_table = builder.make_replay_table(environment_spec)
        self._server = reverb.Server([replay_table], port=None)
        replay_client = reverb.Client(f"localhost:{self._server.port}")

        # The adder is used to insert observations into replay.
        adder = builder.make_adder(replay_client)

        # The dataset provides an interface to sample from replay.
        dataset = builder.make_dataset_iterator(replay_client)

        # Create system architecture
        architecture = DecentralisedActor(
            environment_spec=environment_spec,
            policy_networks=q_networks,
            observation_networks=observation_networks,
            behavior_networks=behavior_networks,
            shared_weights=shared_weights,
        )

        # Augment network architecture by adding mixing layer network.
        networks = MonotonicMixing(
            architecture=architecture,
            state_shape=(1, 1),
            n_agents=5,  # TODO Get this from architecture
            qmix_hidden_dim=2,
        ).create_system()

        # Retrieve networks
        behavior_networks = networks["behaviors"]
        q_networks = networks["policies"]
        target_q_networks = networks["target_policies"]
        observation_networks = networks["observations"]
        mixing_networks = networks["mixing"]

        # Create the actor which defines how we take actions.
        executor = builder.make_executor(behavior_networks, adder)

        trainer_networks = {
            "q_networks": q_networks,
            "target_q_networks": target_q_networks,
            "observation_networks": observation_networks,
            "mixing_networks": mixing_networks,
        }

        # The trainer updates the parameters (and initializes them).
        # TODO label these inputs properly
        trainer = builder.make_trainer(trainer_networks, dataset)

        super().__init__(
            executor=executor,
            trainer=trainer,
            min_observations=batch_size,
            observations_per_step=float(batch_size) / samples_per_insert,
        )
