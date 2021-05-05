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

"""MAPPO system implementation."""
from typing import Callable, Dict, Optional

import reverb
import sonnet as snt
from acme.utils import counting, loggers

from mava import specs
from mava.adders import reverb as reverb_adders

# from mava.components.tf.architectures import DecentralisedValueActorCritic
from mava.systems import system
from mava.systems.tf.mappo import builder


class MAPPO(system.System):

    """MAPPO system.
    This implements a single-process MAPPO system. This is an actor-critic based
    system that generates data via a behavior policy, inserts N-step transitions into
    a replay buffer, and periodically updates the policies of each agent.
    """

    def __init__(
        self,
        environment_spec: specs.MAEnvironmentSpec,
        networks: Dict[str, snt.Module],
        shared_weights: bool = True,
        critic_learning_rate: float = 1e-3,
        policy_learning_rate: float = 1e-3,
        discount: float = 0.99,
        lambda_gae: float = 0.95,
        clipping_epsilon: float = 0.2,
        entropy_cost: float = 0.01,
        baseline_cost: float = 0.5,
        max_abs_reward: Optional[float] = None,
        max_gradient_norm: Optional[float] = None,
        replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE,
        max_queue_size: int = 100_000,
        batch_size: int = 16,
        sequence_length: int = 10,
        sequence_period: int = 5,
        counter: counting.Counter = None,
        logger: loggers.Logger = None,
        checkpoint: bool = True,
    ):

        """Initialize the system.
        Args:
        environment_spec: description of the actions, observations, etc.
        networks: ...
        sequence_length: ...
        sequence_period: ...
        shared_weights: ...
        counter: ...
        entropy_cost: ...
        baseline_cost: ...
        clipping_epsilon: ...
        max_abs_reward: ...
        batch_size: batch size for updates.
        max_queue_size: maximum queue size.
        critic_learning_rate: learning rate for the critic-network update.
        policy_learning_rate: ...
        discount: discount to use for TD updates.
        logger: logger object to be used by learner.
        max_gradient_norm: used for gradient clipping.
        checkpoint: ...
        replay_table_name: string indicating what name to give the replay table."""

        self._builder = builder.MAPPOBuilder(
            config=builder.MAPPOConfig(
                environment_spec=environment_spec,
                shared_weights=shared_weights,
                discount=discount,
                lambda_gae=lambda_gae,
                clipping_epsilon=clipping_epsilon,
                critic_learning_rate=critic_learning_rate,
                policy_learning_rate=policy_learning_rate,
                entropy_cost=entropy_cost,
                baseline_cost=baseline_cost,
                max_abs_reward=max_abs_reward,
                max_gradient_norm=max_gradient_norm,
                replay_table_name=replay_table_name,
                max_queue_size=max_queue_size,
                batch_size=batch_size,
                sequence_length=sequence_length,
                sequence_period=sequence_period,
            ),
            networks=networks,
        )

        # Create a replay server to add data to.
        replay_table = self._builder.make_replay_tables(
            environment_spec=environment_spec
        )
        self._server = reverb.Server([replay_table], port=None)
        replay_client = reverb.Client(f"localhost:{self._server.port}")

        # Create a function to check if we can sample from reverb
        self._can_sample: Callable[[], bool] = lambda: replay_table.can_sample(
            batch_size
        )

        # The adder is used to insert observations into replay.
        adder = self._builder.make_adder(replay_client)

        # The dataset provides an interface to sample from replay.
        dataset = self._builder.make_dataset_iterator(replay_client)

        # Create the executor which defines how we take actions.
        executor = self._builder.make_executor(networks, adder)

        # The learner updates the parameters (and initializes them).
        trainer = self._builder.make_trainer(
            networks, dataset, counter=counter, logger=logger, checkpoint=checkpoint
        )

        super().__init__(
            executor=executor,
            trainer=trainer,
            min_observations=batch_size,
            observations_per_step=batch_size,
        )

    # Overwrite update function to only learn if we can sample
    # from reverb.
    def update(self) -> None:
        learner_step = False
        while self._can_sample():
            self._trainer.step()
            learner_step = True
        if learner_step:
            self._executor.update()
