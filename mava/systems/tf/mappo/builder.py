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

"""MAPPO builder and config."""
import dataclasses
from typing import Dict, Iterator, Optional, Type

import reverb
import sonnet as snt
import tensorflow as tf
from acme import types as acme_types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils
from acme.utils import counting

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.systems.builders import SystemBuilder
from mava.systems.tf.mappo import execution, training


@dataclasses.dataclass
class MAPPOConfig:
    """Configuration options for the MAPPO system
    Args:
        environment_spec: description of the actions, observations, etc.
        shared_weights: ...
        sequence_length: ...
        sequence_period: ...
        entropy_cost: ...
        baseline_cost: ...
        max_abs_reward: ...
        batch_size: batch size for updates.
        max_queue_size: maximum queue size.
        policy_learning_rate:
        critic learning_rate: learning rate for the critic-network update.
        discount: discount to use for TD updates.
        max_gradient_norm: used for gradient clipping.
        replay_table_name: string indicating what name to give the replay table.
    """

    environment_spec: specs.EnvironmentSpec
    sequence_length: int = 10
    sequence_period: int = 5
    shared_weights: bool = False
    discount: float = 0.99
    lambda_gae: float = 0.95
    max_queue_size: int = 100_000
    batch_size: int = 16
    critic_learning_rate: float = 1e-3
    policy_learning_rate: float = 1e-3
    entropy_cost: float = 0.01
    baseline_cost: float = 0.5
    clipping_epsilon: float = 0.2
    max_abs_reward: Optional[float] = None
    max_gradient_norm: Optional[float] = None
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE


class MAPPOBuilder(SystemBuilder):

    """Builder for MAPPO which constructs individual components of the system."""

    """Defines an interface for defining the components of a MARL system.
      Implementations of this interface contain a complete specification of a
      concrete MARL system. An instance of this class can be used to build a
      MARL system which interacts with the environment either locally or in a
      distributed setup.
      """

    def __init__(
        self,
        config: MAPPOConfig,
        networks: Dict[str, snt.Module],
        executer_fn: Type[
            execution.MAPPOFeedForwardExecutor
        ] = execution.MAPPOFeedForwardExecutor,
    ):
        """Args:
        config: Configuration options for the MAPPO system.
        policy_networks: ...
        trainer_fn: ...
        executor_fn: ...
        """

        self._config = config
        self._networks = networks
        self._executor_fn = executer_fn

    def make_replay_tables(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> reverb.Table:
        """Create tables to insert data into."""
        policy_networks = self._networks["policies"]
        agent_specs = environment_spec.get_agent_specs()
        extras_spec: Dict[str, Dict[str, acme_types.NestedArray]] = (
            {"logits": {}, "core_states": {}}
            if self._executor_fn == execution.MAPPORecurrentExecutor
            else {"logits": {}}
        )
        # TODO there is a bit of duplication of work going on here.
        # This should be fixed in the future.
        for agent, spec in agent_specs.items():
            # Get the network key for proper indexing
            network_key = agent.split("_")[0] if self._config.shared_weights else agent

            # Make dummy logits
            # TODO This only supports discreet actions at the moment.
            # We should allow for continuous actions in the future.
            num_actions = agent_specs[agent].actions.num_values
            extras_spec["logits"][network_key] = tf.ones(
                shape=(1, num_actions), dtype=tf.float32
            )

            # Possibly make dummy core state
            if self._executor_fn == execution.MAPPORecurrentExecutor:
                extras_spec["core_states"][network_key] = policy_networks[
                    network_key
                ].initial_state(1)

        # Squeeze the batch dim.
        extras_spec = tf2_utils.squeeze_batch_dim(extras_spec)

        return reverb.Table.queue(
            name=self._config.replay_table_name,
            max_size=self._config.max_queue_size,
            signature=reverb_adders.ParallelSequenceAdder.signature(
                environment_spec, extras_spec=extras_spec
            ),
        )

    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for learning/updating the system.
        Args:
          replay_client: ...
        """
        # Create tensorflow dataset to interface with reverb
        dataset = reverb.ReplayDataset.from_table_signature(
            server_address=replay_client.server_address,
            table=self._config.replay_table_name,
            max_in_flight_samples_per_worker=1,
            sequence_length=self._config.sequence_length,
            emit_timesteps=False,
        )

        # Batching
        dataset = dataset.batch(self._config.batch_size, drop_remainder=True)

        return iter(dataset)

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> Optional[adders.ParallelAdder]:
        """Create an adder which records data generated by the executor/environment.
        Args:
          replay_client: a Reverb client which points to the replay server.
        """
        return reverb_adders.ParallelSequenceAdder(
            client=replay_client,
            period=self._config.sequence_period,
            sequence_length=self._config.sequence_length,
            use_next_extras=False,
        )

    def make_executor(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        adder: Optional[adders.ParallelAdder] = None,
        variable_source: Optional[core.VariableSource] = None,
    ) -> core.Executor:

        """Create an executor instance.
        Args:
            policy_networks: a struct of instance of all the different networks.
            adder: how data is recorded (e.g. added to replay).
            variable_source: a source providing the necessary executor parameters.
        """
        policy_networks = networks["policies"]

        variable_client = None
        if variable_source:
            # Create policy variables.
            variables = {}
            for network_key in policy_networks.keys():
                variables[network_key] = policy_networks[network_key].variables

            # Get new policy variables.
            # TODO Should the variable update period be in the MAPPO config?
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={"policy": variables},
                update_period=50,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        # Create the executor which defines how agents take actions.
        return self._executor_fn(
            policy_networks=policy_networks,
            shared_weights=self._config.shared_weights,
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
            networks: a struct describing the networks needed by the trainer; this can
                be specific to the trainer in question.
            dataset: iterator over samples from replay.
            replay_client: client which allows communication with replay, e.g. in
                order to update priorities.
            counter: a Counter which allows for recording of counts (trainer steps,
                executor steps, etc.) distributed throughout the system.
            logger: Logger object for logging metadata.
            checkpoint: bool controlling whether the trainer checkpoints itself.
        """
        policy_networks = networks["policies"]
        critic_networks = networks["critics"]

        # The learner updates the parameters (and initializes them).
        trainer = training.MAPPOTrainer(
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            dataset=dataset,
            shared_weights=self._config.shared_weights,
            critic_learning_rate=self._config.critic_learning_rate,
            policy_learning_rate=self._config.policy_learning_rate,
            discount=self._config.discount,
            lambda_gae=self._config.lambda_gae,
            entropy_cost=self._config.entropy_cost,
            baseline_cost=self._config.baseline_cost,
            clipping_epsilon=self._config.clipping_epsilon,
            max_abs_reward=self._config.max_abs_reward,
            max_gradient_norm=self._config.max_gradient_norm,
            counter=counter,
            logger=logger,
        )
        return trainer
