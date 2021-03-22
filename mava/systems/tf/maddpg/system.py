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


from typing import Dict

import reverb
import sonnet as snt
from acme import datasets
from acme.adders import reverb as adders
from acme.utils import counting, loggers

from mava import specs
from mava.components.tf.architectures import CentralisedActorCritic
from mava.systems import system
from mava.systems.tf import executors
from mava.systems.tf.maddpg import training


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
        """Initialize the agent.
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

        # Create a replay server to add data to. This uses no limiter behavior in
        # order to allow the Agent interface to handle it.
        replay_table = reverb.Table(
            name="replay_table_name",
            sampler=reverb.selectors.Uniform(),
            remover=reverb.selectors.Fifo(),
            max_size=max_replay_size,
            rate_limiter=reverb.rate_limiters.MinSize(1),
            signature=adders.NStepTransitionAdder.signature(environment_spec),
        )
        self._server = reverb.Server([replay_table], port=None)

        # The adder is used to insert observations into replay.
        address = f"localhost:{self._server.port}"
        adder = adders.NStepTransitionAdder(
            priority_fns={replay_table_name: lambda x: 1.0},
            client=reverb.Client(address),
            n_step=n_step,
            discount=discount,
        )

        # The dataset provides an interface to sample from replay.
        dataset = datasets.make_reverb_dataset(
            table=replay_table_name,
            server_address=address,
            batch_size=batch_size,
            prefetch_size=prefetch_size,
        )

        networks = CentralisedActorCritic(
            environment_spec=environment_spec,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            observation_networks=observation_networks,
            shared_weights=shared_weights,
        ).create_system()

        # Create the actor which defines how we take actions.
        executor = executors.FeedForwardExecutor(
            policy_networks=networks["policies"],
            shared_weights=shared_weights,
            adder=adder,
        )

        # Create optimizers.
        policy_optimizer = snt.optimizers.Adam(learning_rate=1e-4)
        critic_optimizer = snt.optimizers.Adam(learning_rate=1e-4)

        agents = environment_spec.get_agent_ids()
        agent_types = environment_spec.get_agent_types()

        # The learner updates the parameters (and initializes them).
        trainer = training.MADDPGTrainer(
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

        super().__init__(
            executor=executor,
            trainer=trainer,
            min_observations=max(batch_size, min_replay_size),
            observations_per_step=float(batch_size) / samples_per_insert,
        )
