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

"""MAPPO system builder implementation."""

import dataclasses
from typing import Dict, Iterator, List, Optional, Type, Union

import reverb
import sonnet as snt
import tensorflow as tf
from acme import datasets
from acme import types as acme_types
from acme.tf import utils as tf2_utils
from acme.tf import variable_utils
from acme.utils import counting

from mava import adders, core, specs, types
from mava.adders import reverb as reverb_adders
from mava.systems.tf.mappo import execution, training
from mava.wrappers import DetailedTrainerStatistics


@dataclasses.dataclass
class MAPPOConfig:
    """Configuration options for the MAPPO system
    Args:
        environment_spec: description of the action and observation spaces etc. for
            each agent in the system.
        policy_optimizer: optimizer(s) for updating policy networks.
        critic_optimizer: optimizer for updating critic networks.
        agent_net_keys: (dict, optional): specifies what network each agent uses.
            Defaults to {}.
        checkpoint_minute_interval (int): The number of minutes to wait between
            checkpoints.
        sequence_length: recurrent sequence rollout length.
        sequence_period: consecutive starting points for overlapping rollouts across a
            sequence.
        discount: discount to use for TD updates.
        lambda_gae: scalar determining the mix of bootstrapping vs further accumulation
            of multi-step returns at each timestep. See `High-Dimensional Continuous
            Control Using Generalized Advantage Estimation` for more information.
        max_queue_size: maximum number of items in the queue.
        executor_variable_update_period: the rate at which executors sync their
            paramters with the trainer.
        batch_size: batch size for updates.
        entropy_cost: contribution of entropy regularization to the total loss.
        baseline_cost: contribution of the value loss to the total loss.
        clipping_epsilon: Hyper-parameter for clipping in the policy objective. Roughly:
            how far can the new policy go from the old policy while still profiting?
            The new policy can still go farther than the clip_ratio says, but it doesn’t
            help on the objective anymore.
        max_gradient_norm: value to specify the maximum clipping value for the gradient
            norm during optimization.
        checkpoint: boolean to indicate whether to checkpoint models.
        checkpoint_subpath: subdirectory specifying where to store checkpoints.
        replay_table_name: string indicating what name to give the replay table.
    """

    environment_spec: specs.EnvironmentSpec
    policy_optimizer: Union[snt.Optimizer, Dict[str, snt.Optimizer]]
    critic_optimizer: snt.Optimizer
    agent_net_keys: Dict[str, str]
    checkpoint_minute_interval: int
    sequence_length: int = 10
    sequence_period: int = 5
    discount: float = 0.99
    lambda_gae: float = 0.95
    max_queue_size: int = 100_000
    executor_variable_update_period: int = 100
    batch_size: int = 32
    entropy_cost: float = 0.01
    baseline_cost: float = 0.5
    clipping_epsilon: float = 0.1
    max_gradient_norm: Optional[float] = None
    checkpoint: bool = True
    checkpoint_subpath: str = "~/mava/"
    replay_table_name: str = reverb_adders.DEFAULT_PRIORITY_TABLE


class MAPPOBuilder:
    """Builder for MAPPO which constructs individual components of the system."""

    def __init__(
        self,
        config: MAPPOConfig,
        trainer_fn: Type[training.MAPPOTrainer] = training.MAPPOTrainer,
        executor_fn: Type[core.Executor] = execution.MAPPOFeedForwardExecutor,
    ):
        """Initialise the system.

        Args:
            config (MAPPOConfig): system configuration specifying hyperparameters and
                additional information for constructing the system.
            trainer_fn (Type[training.MAPPOTrainer], optional): Trainer
                function, of a correpsonding type to work with the selected system
                architecture. Defaults to training.MAPPOTrainer.
            executor_fn (Type[core.Executor], optional): Executor function, of a
                corresponding type to work with the selected system architecture.
                Defaults to execution.MAPPOFeedForwardExecutor.
        """

        self._config = config
        self._agents = self._config.environment_spec.get_agent_ids()
        self._agent_types = self._config.environment_spec.get_agent_types()
        self._trainer_fn = trainer_fn
        self._executor_fn = executor_fn

    def make_replay_tables(
        self,
        environment_spec: specs.MAEnvironmentSpec,
    ) -> List[reverb.Table]:
        """Create tables to insert data into.

        Args:
            environment_spec (specs.MAEnvironmentSpec): description of the action and
                observation spaces etc. for each agent in the system.

        Returns:
            List[reverb.Table]: a list of data tables for inserting data.
        """

        agent_specs = environment_spec.get_agent_specs()
        extras_spec: Dict[str, Dict[str, acme_types.NestedArray]] = {"log_probs": {}}

        for agent, spec in agent_specs.items():
            # Make dummy log_probs
            extras_spec["log_probs"][agent] = tf.ones(shape=(1,), dtype=tf.float32)

        # Squeeze the batch dim.
        extras_spec = tf2_utils.squeeze_batch_dim(extras_spec)

        replay_table = reverb.Table.queue(
            name=self._config.replay_table_name,
            max_size=self._config.max_queue_size,
            signature=reverb_adders.ParallelSequenceAdder.signature(
                environment_spec,
                sequence_length=self._config.sequence_length,
                extras_spec=extras_spec,
            ),
        )

        return [replay_table]

    def make_dataset_iterator(
        self,
        replay_client: reverb.Client,
    ) -> Iterator[reverb.ReplaySample]:
        """Create a dataset iterator to use for training/updating the system.

        Args:
            replay_client (reverb.Client): Reverb Client which points to the
                replay server.

        Returns:
            [type]: dataset iterator.

        Yields:
            Iterator[reverb.ReplaySample]: data samples from the dataset.
        """

        # Create tensorflow dataset to interface with reverb
        dataset = datasets.make_reverb_dataset(
            server_address=replay_client.server_address,
            batch_size=self._config.batch_size,
        )

        return iter(dataset)

    def make_adder(
        self,
        replay_client: reverb.Client,
    ) -> Optional[adders.ParallelAdder]:
        """Create an adder which records data generated by the executor/environment.

        Args:
            replay_client (reverb.Client): Reverb Client which points to the
                replay server.

        Raises:
            NotImplementedError: unknown executor type.

        Returns:
            Optional[adders.ParallelAdder]: adder which sends data to a replay buffer.
        """

        return reverb_adders.ParallelSequenceAdder(
            client=replay_client,
            period=self._config.sequence_period,
            sequence_length=self._config.sequence_length,
            use_next_extras=False,
        )

    def make_executor(
        self,
        policy_networks: Dict[str, snt.Module],
        adder: Optional[adders.ParallelAdder] = None,
        variable_source: Optional[core.VariableSource] = None,
    ) -> core.Executor:

        """Create an executor instance.

        Args:
            policy_networks (Dict[str, snt.Module]): policy networks for each agent in
                the system.
            adder (Optional[adders.ParallelAdder], optional): adder to send data to
                a replay buffer. Defaults to None.
            variable_source (Optional[core.VariableSource], optional): variables server.
                Defaults to None.

        Returns:
            core.Executor: system executor, a collection of agents making up the part
                of the system generating data by interacting the environment.
        """

        variable_client = None
        if variable_source:
            # Create policy variables.
            variables = {
                network_key: policy_networks[network_key].variables
                for network_key in policy_networks
            }

            # Get new policy variables.
            # TODO Should the variable update period be in the MAPPO config?
            variable_client = variable_utils.VariableClient(
                client=variable_source,
                variables={"policy": variables},
                update_period=self._config.executor_variable_update_period,
            )

            # Make sure not to use a random policy after checkpoint restoration by
            # assigning variables before running the environment loop.
            variable_client.update_and_wait()

        # Create the executor which defines how agents take actions.
        return self._executor_fn(
            policy_networks=policy_networks,
            agent_net_keys=self._config.agent_net_keys,
            variable_client=variable_client,
            adder=adder,
        )

    def make_trainer(
        self,
        networks: Dict[str, Dict[str, snt.Module]],
        dataset: Iterator[reverb.ReplaySample],
        counter: Optional[counting.Counter] = None,
        logger: Optional[types.NestedLogger] = None,
    ) -> core.Trainer:
        """Create a trainer instance.

        Args:
            networks (Dict[str, Dict[str, snt.Module]]): system networks.
            dataset (Iterator[reverb.ReplaySample]): dataset iterator to feed data to
                the trainer networks.
            counter (Optional[counting.Counter], optional): a Counter which allows for
                recording of counts, e.g. trainer steps. Defaults to None.
            logger (Optional[types.NestedLogger], optional): Logger object for logging
                metadata. Defaults to None.

        Returns:
            core.Trainer: system trainer, that uses the collected data from the
                executors to update the parameters of the agent networks in the system.
        """

        agents = self._agents
        agent_types = self._agent_types
        agent_net_keys = self._config.agent_net_keys

        observation_networks = networks["observations"]
        policy_networks = networks["policies"]
        critic_networks = networks["critics"]

        # The learner updates the parameters (and initializes them).
        trainer = self._trainer_fn(
            agents=agents,
            agent_types=agent_types,
            observation_networks=observation_networks,
            policy_networks=policy_networks,
            critic_networks=critic_networks,
            dataset=dataset,
            agent_net_keys=agent_net_keys,
            critic_optimizer=self._config.critic_optimizer,
            policy_optimizer=self._config.policy_optimizer,
            discount=self._config.discount,
            lambda_gae=self._config.lambda_gae,
            entropy_cost=self._config.entropy_cost,
            baseline_cost=self._config.baseline_cost,
            clipping_epsilon=self._config.clipping_epsilon,
            max_gradient_norm=self._config.max_gradient_norm,
            counter=counter,
            logger=logger,
            checkpoint_minute_interval=self._config.checkpoint_minute_interval,
            checkpoint=self._config.checkpoint,
            checkpoint_subpath=self._config.checkpoint_subpath,
        )

        trainer = DetailedTrainerStatistics(  # type: ignore
            trainer, metrics=["policy_loss", "critic_loss"]
        )

        return trainer
