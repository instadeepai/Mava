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

import dataclasses
from typing import Dict, Iterator, List, Optional, Type

import reverb
import sonnet as snt
import tensorflow as tf
from acme import datasets
from acme.utils import counting

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.systems.builders import SystemBuilder
from mava.systems.tf import executors
from mava.systems.tf.madqn import training
from mava.wrappers import DetailedTrainerStatistics


@dataclasses.dataclass
class MADQNConfig:
    """Configuration options for the MADDPG system.
    Args:
            environment_spec: description of the actions, observations, etc.
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
            replay_table_name: string indicating what name to give the replay table."""

    environment_spec: specs.MAEnvironmentSpec
    epsilon: tf.Variable
    shared_weights: bool
    target_update_period: int
    clipping: bool
    min_replay_size: int
    max_replay_size: int
    samples_per_insert: Optional[float]
    prefetch_size: int
    batch_size: int
    n_step: int
    discount: float
    policy_optimizer: snt.Optimizer
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE


class MADQNBuilder(SystemBuilder):
    """Builder for MADDPG which constructs individual components of the system."""

    """Defines an interface for defining the components of an RL system.
      Implementations of this interface contain a complete specification of a
      concrete RL system. An instance of this class can be used to build an
      RL system which interacts with the environment either locally or in a
      distributed setup.
      """

    def __init__(
        self,
        config: MADQNConfig,
        trainer_fn: Type[training.IDQNTrainer] = training.IDQNTrainer,
        executer_fn: Type[core.Executor] = executors.FeedForwardExecutor,
    ):
        """Args:
        _config: Configuration options for the MADDPG system.
        _trainer_fn: Trainer module to use."""
        self._config = config
        self._trainer_fn = trainer_fn
        self._executor_fn = executer_fn

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
    ) -> List[reverb.Table]:
        if self._config.samples_per_insert is None:
            # We will take a samples_per_insert ratio of None to mean that there is
            # no limit, i.e. this only implies a min size limit.
            limiter = reverb.rate_limiters.MinSize(self._config.min_replay_size)

        else:
            # Create enough of an error buffer to give a 10% tolerance in rate.
            samples_per_insert_tolerance = 0.1 * self._config.samples_per_insert
            error_buffer = self._config.min_replay_size * samples_per_insert_tolerance
            limiter = reverb.rate_limiters.SampleToInsertRatio(
                min_size_to_sample=self._config.min_replay_size,
                samples_per_insert=self._config.samples_per_insert,
                error_buffer=error_buffer,
            )

        replay_table = reverb.Table(
            name=self._config.replay_table_name,
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=limiter,
            signature=reverb_adders.ParallelNStepTransitionAdder.signature(
                environment_spec
            ),
        )

        return [replay_table]

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
        return self._executor_fn(
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
        logger: Optional[types.NestedLogger] = None,
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
        q_networks = networks["values"]
        target_q_networks = networks["target_values"]

        agents = self._config.environment_spec.get_agent_ids()
        agent_types = self._config.environment_spec.get_agent_types()

        # Create optimizers.
        optimizer = self._config.policy_optimizer

        # The learner updates the parameters (and initializes them).
        trainer = self._trainer_fn(
            agents=agents,
            agent_types=agent_types,
            q_networks=q_networks,
            target_q_networks=target_q_networks,
            epsilon=self._config.epsilon,
            shared_weights=self._config.shared_weights,
            optimizer=optimizer,
            target_update_period=self._config.target_update_period,
            clipping=self._config.clipping,
            dataset=dataset,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
        )

        trainer = DetailedTrainerStatistics(trainer)  # type: ignore

        return trainer
