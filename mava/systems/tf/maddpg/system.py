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

import copy
import dataclasses
from typing import Dict, Iterator, List, Optional, Union

import numpy as np
import reverb
import sonnet as snt
import tensorflow as tf
from acme import datasets, specs, types
from acme.tf import networks as acme_networks
from acme.tf import utils as tf2_utils
from acme.utils import counting, loggers

from mava import adders, core
from mava.systems import system
from mava.systems.tf import executors
from mava.systems.tf.builders import SystemBuilder
from mava.systems.tf.maddpg import training

NestedLogger = Union[loggers.Logger, Dict[str, loggers.Logger]]


@dataclasses.dataclass
class MADDPGConfig:
    """Configuration options for the MADDPG system.
    Args:
            environment_spec: description of the actions, observations, etc.
            policy_network: the online (optimized) policy.
            critic_network: the online critic.
            observation_network: optional network to transform the observations before
              they are fed into any network.
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
            logger: logger object to be used by learner.
            counter: counter object used to keep track of steps.
            checkpoint: boolean indicating whether to checkpoint the learner.
            replay_table_name: string indicating what name to give the replay table."""

    agents: List[str]
    agent_types: List[str]
    environment_spec: specs.EnvironmentSpec
    policy_networks: Dict[str, snt.Module]
    critic_networks: Dict[str, snt.Module]
    observation_networks: Dict[str, snt.Module]
    shared_weights: bool = False
    discount: float = 0.99
    batch_size: int = 256
    prefetch_size: int = 4
    target_update_period: int = 100
    min_replay_size: int = 1000
    max_replay_size: int = 1000000
    samples_per_insert: float = 32.0
    n_step: int = 5
    sigma: float = 0.3
    clipping: bool = True
    logger: loggers.Logger = None
    counter: counting.Counter = None
    checkpoint: bool = True
    replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE


class MADDPGBuilder(SystemBuilder):
    """Builder for D4PG which constructs individual components of the agent."""

    """Defines an interface for defining the components of an RL system.
      Implementations of this interface contain a complete specification of a
      concrete RL system. An instance of this class can be used to build an
      RL system which interacts with the environment either locally or in a
      distributed setup.
      """

    def __init__(self, config: MADDPGConfig):
        self._config = config

    def make_replay_tables(
        self,
        environment_spec: specs.EnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into."""
        return reverb.Table(
            name="replay_table_name",
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=self._config.max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=adders.NStepTransitionAdder.signature(environment_spec),
        )

    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the agent."""
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
    ) -> Optional[adders.Adder]:
        """Create an adder which records data generated by the executor/environment.
        Args:
          replay_client: Reverb Client which points to the replay server.
        """
        return adders.NStepTransitionAdder(
            priority_fns={self._config.replay_table_name: lambda x: 1.0},
            client=replay_client,
            n_step=self._config.n_step,
            discount=self._config.discount,
        )

    def make_executor(
        self,
        policy_networks,
        adder: Optional[adders.Adder] = None,
        variable_source: Optional[core.VariableSource] = None,
    ) -> core.Executor:
        """Create an executer instance.
        Args:
          policy_networks: A struct of instance of all the different policy networks;
           this should be a callable
            which takes as input observations and returns actions.
          adder: How data is recorded (e.g. added to replay).
          variable_source: A source providing the necessary actor parameters.
        """
        shared_weights = self._config.shared_weights

        # Create the actor which defines how we take actions.
        return executors.FeedForwardExecutor(
            policy_networks=policy_networks,
            shared_weights=shared_weights,
            variable_source=variable_source,
            adder=adder,
        )

    def make_trainer(
        self,
        networks,
        dataset: Iterator[reverb.ReplaySample],
        # replay_client: Optional[reverb.Client] = None,
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
          replay_client: client which allows communication with replay, e.g. in
            order to update priorities.
          counter: a Counter which allows for recording of counts (learner steps,
            actor steps, etc.) distributed throughout the agent.
          logger: Logger object for logging metadata.
          checkpoint: bool controlling whether the learner checkpoints itself.
        """
        observation_networks = self._config.observation_networks

        agents = self._config.agents
        agent_types = self._config.agent_types
        shared_weights = self._config.shared_weights
        clipping = self._config.clipping
        discount = self._config.discount
        target_update_period = self._config.target_update_period

        (
            policy_networks,
            critic_networks,
            target_policy_networks,
            target_critic_networks,
            target_observation_networks,
        ) = networks

        # Create optimizers.
        policy_optimizer = snt.optimizers.Adam(learning_rate=1e-4)
        critic_optimizer = snt.optimizers.Adam(learning_rate=1e-4)

        # The learner updates the parameters (and initializes them).
        trainer = training.MADDPGTrainer(
            agents=agents,
            agent_types=agent_types,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            observation_networks=observation_networks,
            target_policy_networks=target_policy_networks,
            target_critic_networks=target_critic_networks,
            target_observation_networks=target_observation_networks,
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
        agents: List[str],
        agent_types: List[str],
        environment_spec: specs.EnvironmentSpec,
        policy_networks: Dict[str, snt.Module],
        critic_networks: Dict[str, snt.Module],
        observation_networks: Dict[str, snt.Module],
        shared_weights: bool = False,
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
        replay_table_name: str = adders.DEFAULT_PRIORITY_TABLE,
    ):
        """Initialize the system.
        Args:
          environment_spec: description of the actions, observations, etc.
          policy_network: the online (optimized) policy.
          critic_network: the online critic.
          observation_network: optional network to transform the observations before
            they are fed into any network.
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
          logger: logger object to be used by learner.
          counter: counter object used to keep track of steps.
          checkpoint: boolean indicating whether to checkpoint the learner.
          replay_table_name: string indicating what name to give the replay table.
        """

        builder = MADDPGBuilder(
            MADDPGConfig(
                agents=agents,
                agent_types=agent_types,
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
            )
        )

        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        replay_table = builder.make_replay_tables()
        self._server = reverb.Server([replay_table], port=None)
        replay_client = reverb.Client(f"localhost:{self._server.port}")

        # The adder is used to insert observations into replay.
        adder = builder.make_adder(replay_client)

        # The dataset provides an interface to sample from replay.
        dataset = builder.make_dataset_iterator(replay_client)

        # Create networks
        # TODO: Should this not be fed in through the policy_networks variable instead of _config?
        #  e.g. policy_networks = (policy_networks, critic_networks)
        #  Should the critic network even be in the make_executor function?

        n_agents = len(agents)
        behavior_networks = {}
        target_policy_networks = {}
        target_critic_networks = {}
        target_observation_networks = {}

        if observation_networks is None:
            observation_networks = {}
            create_observation_networks = True

        agent_keys = self._agent_type if shared_weights else self._agents

        for agent_key in agent_keys:
            # Make sure observation network is a Sonnet Module.
            if create_observation_networks:
                observation_network: types.TensorTransformation = tf.identity
                observation_network = tf2_utils.to_sonnet_module(observation_network)
                observation_networks[agent_key] = observation_network

            # Get observation and action specs.
            act_spec = environment_spec[agent_key].actions
            obs_spec = environment_spec[agent_key].observations
            emb_spec = tf2_utils.create_variables(
                observation_networks[agent_key], [obs_spec]
            )
            critic_state_spec = np.tile(obs_spec, n_agents)
            critic_act_spec = np.tile(act_spec, n_agents)

            # Create target networks.
            target_policy_network = copy.deepcopy(policy_networks[agent_key])
            target_critic_network = copy.deepcopy(critic_networks[agent_key])
            target_observation_network = copy.deepcopy(observation_networks[agent_key])

            target_policy_networks[agent_key] = target_policy_network
            target_critic_networks[agent_key] = target_critic_network
            target_observation_networks[agent_key] = target_observation_network

            # Create the behavior policy.
            behavior_network = snt.Sequential(
                [
                    observation_network,
                    policy_networks[agent_key],
                    acme_networks.ClippedGaussian(sigma),
                    acme_networks.ClipToSpec(act_spec),
                ]
            )
            behavior_networks[agent_key] = behavior_network

            # Create variables.
            tf2_utils.create_variables(policy_networks[agent_key], [emb_spec])
            tf2_utils.create_variables(
                critic_networks[agent_key], [critic_state_spec, critic_act_spec]
            )
            tf2_utils.create_variables(target_policy_network, [emb_spec])
            tf2_utils.create_variables(
                target_critic_network, [critic_state_spec, critic_act_spec]
            )
            tf2_utils.create_variables(target_observation_network, [obs_spec])

        # Create the executor
        policy_networks = behavior_networks
        executor = builder.make_executor(policy_networks, adder)

        # Create the trainer
        networks = (
            policy_networks,
            critic_networks,
            target_policy_networks,
            target_critic_networks,
            target_observation_networks,
        )
        trainer = builder.make_trainer(networks, dataset, counter, logger, checkpoint)

        super().__init__(
            executor=executor,
            trainer=trainer,
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=float(batch_size) / samples_per_insert,
        )
