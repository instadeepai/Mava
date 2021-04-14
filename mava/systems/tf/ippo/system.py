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
from typing import Dict, Iterator, Optional, Type

import reverb
import sonnet as snt
from acme import datasets
from acme.tf import variable_utils
from acme.utils import counting, loggers

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders

# from mava.components.tf.architectures import DecentralisedActorCritic
# from mava.systems import system
from mava.systems.builders import SystemBuilder
from mava.systems.tf import executors
from mava.systems.tf.ippo import training


@dataclasses.dataclass
class IPPOConfig:
    """Configuration options for the IPPO system.
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

    # Network stuff
    environment_spec: specs.MAEnvironmentSpec
    policy_networks: Dict[str, snt.Module]
    critic_networks: Dict[str, snt.Module]
    observation_networks: Dict[str, snt.Module]
    shared_weights: bool = True
    target_update_period: int = 100

    # PPO stuff
    discount: float = 0.99
    lambda_gae: float = 0.95
    clipping_epsilon: float = 0.2
    entropy_cost: float = 0.01
    baseline_cost: float = 0.5

    # Reverb stuff
    batch_size: int = 32
    prefetch_size: int = 4
    max_queue_size: int = 100_000
    samples_per_insert: float = 32.0
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE
    n_step = 5

    # Gradient clipping
    clipping: bool = True

    # Extra stuff
    logger: loggers.Logger = None
    counter: counting.Counter = None
    checkpoint: bool = True


class IPPOBuilder(SystemBuilder):
    """Builder for IPPO which constructs individual components of the system."""

    """Defines an interface for defining the components of an RL system.
      Implementations of this interface contain a complete specification of a
      concrete RL system. An instance of this class can be used to build an
      RL system which interacts with the environment either locally or in a
      distributed setup.
      """

    def __init__(
        self,
        config: IPPOConfig,
        trainer_fn: Type[training.IPPOTrainer] = training.IPPOTrainer,
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

    def make_replay_table(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> reverb.Table.queue:
        """Create tables to insert data into."""
        return reverb.Table.queue(
            name=self._config.replay_table_name,
            max_size=self._config.max_queue_size,
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
            priority_fns=None,  # {self._config.replay_table_name: lambda x: 1.0},
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

            # Update variables
            # TODO: Is this needed? Probably not because
            #  in acme they only update policy.variables.
            # for agent in agent_keys:
            #     policy_networks[agent].variables = variables[agent]

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        # TODO may need a custom executor
        # Create the actor which defines how we take actions.
        return executors.FeedForwardExecutor(
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
        entropy_cost = self._config.entropy_cost
        baseline_cost = self._config.baseline_cost
        lambda_gae = self._config.lambda_gae
        clipping_epsilon = self._config.clipping_epsilon
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
            # target_policy_networks=networks["target_policies"],
            # target_critic_networks=networks["target_critics"],
            # target_observation_networks=networks["target_observations"],
            shared_weights=shared_weights,
            policy_optimizer=policy_optimizer,
            critic_optimizer=critic_optimizer,
            entropy_cost=entropy_cost,
            baseline_cost=baseline_cost,
            lambda_gae=lambda_gae,
            clipping_epsilon=clipping_epsilon,
            clipping=clipping,
            discount=discount,
            target_update_period=target_update_period,
            dataset=dataset,
            counter=counter,
            logger=logger,
            checkpoint=checkpoint,
        )
        return trainer
